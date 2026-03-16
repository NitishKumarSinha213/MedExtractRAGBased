package RagExtractor;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.pdfbox.ApachePdfBoxDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.Result;
import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/SearchRAG")
@CrossOrigin("http://localhost:3000")
public class RagExtractor {

    // Define the interface with strict system instructions
    public interface MedicalAssistant {
        @SystemMessage({
//                "You are a professional medical data extractor.",
//                "Use ONLY the provided context from the uploaded PDF to answer the query.",
//                "If the information is not in the document, say 'Information not found in the provided record.'",
//                "Do not use your own internal knowledge to provide hypothetical medical data."
                "Role: You are a Specialized Medical Data Extraction Assistant." +
                        " Your objective is to parse clinical documentation " +
                        "(e.g., EHR notes, pathology reports, discharge summaries)" +
                        " into structured data with 100% clinical fidelity and zero conversational filler." +
                        "Context: You will be processing sensitive medical information. " +
                        "The extracted data must reflect the literal content of the medical record without interpretation." +
                        " You must distinguish between \"Negative/Absent\" findings and \"Not Found\" data points to ensure " +
                        "the clinical record remains accurate for diagnostic history." +
                        "Information: * Clinical Accuracy: Extract medical terms exactly as written. " +
                        "If standard codes (e.g., ICD-10, SNOMED) are requested, do not guess;" +
                        " only map them if the relationship is explicit.Status Differentiation:" +
                        " Distinguish between Current Diagnosis, History of, and Family History." +
                        "Missing Fields: If a specific field is not mentioned at all in the text, return exactly: Not Found." +
                        " If the text explicitly states a symptom is absent, return: Absent." +
                        "Metric Precision: Always include units of measurement (e.g., $mg/dL$, $120/80\\ mmHg$) and the" +
                        " timestamp of the reading if available." +
                        "Justification: When justification is required, provide a maximum of 1–3 words citing the" +
                        " source section (e.g., \"Vitals section,\" \"Lab results,\" \"Plan of care\")." +
                        "Note: * Zero Interpretation: Do not \"correct\" a clinician’s spelling or" +
                        " provide medical advice/summaries.Prohibited Content:" +
                        " Do NOT use headings like \"Extracted data,\" \"Summary,\" or include validation markers like" +
                        " \"(Correct)\" or \"(Verified).\"" +
                        "No Overtalking: Provide the raw data points directly. " +
                        "No introductory greetings or concluding remarks." +
                        "HIPAA Awareness: Do not generate or append any PII (Personally Identifiable Information) that is" +
                        " not explicitly present in the source text." +
                        "Only search/extract what the user instruction is asking to do. Do not extract other data. Keep it" +
                        "precise and limited." +
                        "You can reach to the embedding model for attaching codes or other medical terms if you are confident about" +
                        "the accuracy of that data"
        })
        Result<String> chat(String message);
    }

    private final EmbeddingModel embeddingModel;
    private final OllamaChatModel chatModel;

    public RagExtractor() {
        // Initialize the models once during startup
        this.embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        this.chatModel = OllamaChatModel.builder()
                .baseUrl("http://localhost:11434")
                .modelName("qwen2.5:7b")
                .build();
    }

    @PostMapping(consumes = "multipart/form-data")
    public ResponseEntity<Map<String, Object>> runUserQuery(
            @RequestParam("userQuery") String userQuery,
            @RequestParam("file") MultipartFile file) throws IOException {

        // 1. Create a FRESH, temporary store for this request only
        EmbeddingStore<TextSegment> ephemeralStore = new InMemoryEmbeddingStore<>();

        // 2. Process the uploaded file
        Path tempFile = Files.createTempFile("current-patient", ".pdf");
        file.transferTo(tempFile);

        try (PDDocument pdDocument = PDDocument.load(tempFile.toFile())) {
            PDFTextStripper stripper = new PDFTextStripper();
            stripper.setSortByPosition(true); // Forces logical reading order from top to bottom
            String fullText = stripper.getText(pdDocument);

            // Create the LangChain4j document from the explicitly stripped text
            Document document = Document.from(fullText);

            DocumentSplitter childSplitter = DocumentSplitters.recursive(300, 50);
            Map<String, String> parentStore = new HashMap<>();
            // 3. Chunk the document for better accuracy
            List<TextSegment> segments = DocumentSplitters.recursive(3000, 200).split(document);
            for (TextSegment parent : segments) {
                String parentId = UUID.randomUUID().toString();
                parentStore.put(parentId, parent.text()); // Store the big text for later

                Document tempDocument = Document.from(parent.text(), parent.metadata());
                // Split parent into smaller children
                List<TextSegment> children = childSplitter.split(tempDocument);

                for (TextSegment child : children) {
                    // Tie child to parent in metadata
                    child.metadata().put("parent_id", parentId);
                    ephemeralStore.add(embeddingModel.embed(child).content(), child);
                }

                // Store parent text in a simple Map or DB
                parentStore.put(parentId, parent.text());
            }
            ephemeralStore.addAll(embeddingModel.embedAll(segments).content(), segments);

            // 4. Create a temporary retriever and assistant for this specific store
            ContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
                    .embeddingStore(ephemeralStore)
                    .embeddingModel(embeddingModel)
                    .maxResults(3)
                    .build();

            MedicalAssistant assistant = AiServices.builder(MedicalAssistant.class)
                    .chatLanguageModel(chatModel)
                    .contentRetriever(retriever)
                    .build();

            // 5. Execute search
            Result<String> result = assistant.chat(userQuery);
            System.out.println("Retrieved Segments: " + result.sources().stream()
                    .map(s -> s.textSegment().text())
                    .collect(Collectors.joining(" | ")));

            // 6. Format response with sources
            Map<String, Object> response = new HashMap<>();
            response.put("queryResult", result.content());
            response.put("sources", result.sources().stream()
                    .map(s -> s.textSegment().text())
                    .collect(Collectors.toList()));

            return ResponseEntity.ok(response);

        } finally {
            Files.deleteIfExists(tempFile);
            // ephemeralStore goes out of scope and is cleared by Garbage Collector
        }
    }
}