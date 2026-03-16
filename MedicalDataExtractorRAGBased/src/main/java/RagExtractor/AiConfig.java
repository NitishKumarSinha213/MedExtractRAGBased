package RagExtractor;

import dev.langchain4j.model.huggingface.HuggingFaceEmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.time.Duration;

@Configuration
public class AiConfig {

    @Value("${huggingface.access-token}")
    private String token;

    @Value("${huggingface.model-id}")
    private String modelId;

    @Bean
    public EmbeddingModel embeddingModel() {
        return HuggingFaceEmbeddingModel.builder()
                .accessToken(token)
                .modelId(modelId)
                .timeout(Duration.ofSeconds(60))
                .build();
    }
}