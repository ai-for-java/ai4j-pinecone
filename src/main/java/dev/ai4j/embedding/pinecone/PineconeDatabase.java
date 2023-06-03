package dev.ai4j.embedding.pinecone;

import com.google.protobuf.Struct;
import com.google.protobuf.Value;
import dev.ai4j.embedding.Embedding;
import dev.ai4j.embedding.VectorDatabase;
import io.pinecone.PineconeClient;
import io.pinecone.PineconeClientConfig;
import io.pinecone.PineconeConnection;
import io.pinecone.PineconeConnectionConfig;
import io.pinecone.proto.*;
import lombok.Builder;

import java.util.Collection;
import java.util.List;
import java.util.UUID;

import static java.util.Collections.singletonList;
import static java.util.stream.Collectors.toList;

public class PineconeDatabase implements VectorDatabase {

    private static final String DEFAULT_NAMESPACE = "default";
    private static final String METADATA_ORIGINAL_TEXT = "text";

    PineconeConnection connection;
    String nameSpace;

    @Builder
    public PineconeDatabase(String apiKey, String environment, String projectName, String index, String nameSpace) {

        PineconeClientConfig configuration = new PineconeClientConfig()
                .withApiKey(apiKey)
                .withEnvironment(environment)
                .withProjectName(projectName);

        PineconeClient pineconeClient = new PineconeClient(configuration);

        PineconeConnectionConfig connectionConfig = new PineconeConnectionConfig()
                .withIndexName(index);

        this.connection = pineconeClient.connect(connectionConfig); // TODO close
        this.nameSpace = nameSpace == null ? DEFAULT_NAMESPACE : nameSpace;
    }

    @Override
    public void persist(Embedding embedding) {
        persist(singletonList(embedding));
    }

    @Override
    public void persist(Iterable<Embedding> embeddings) {

        UpsertRequest.Builder upsertRequestBuilder = UpsertRequest.newBuilder()
                .setNamespace(nameSpace);

        embeddings.forEach(embedding -> {

            Value originalText = Value.newBuilder()
                    .setStringValue(embedding.contents())
                    .build();

            Struct vectorMetadata = Struct.newBuilder()
                    .putFields(METADATA_ORIGINAL_TEXT, originalText)
                    .build();

            Vector vector = Vector.newBuilder()
                    .setId(createUniqueId())
                    .addAllValues(toFloats(embedding))
                    .setMetadata(vectorMetadata)
                    .build();

            upsertRequestBuilder.addVectors(vector);
        });

        connection.getBlockingStub().upsert(upsertRequestBuilder.build());

        // TODO verify that all embeddings are persisted ?
    }

    @Override
    public List<Embedding> findRelated(Embedding embedding, int maxResults) {

        QueryVector queryVector = QueryVector
                .newBuilder()
                .addAllValues(toFloats(embedding))
                .setTopK(maxResults)
                .setNamespace(nameSpace)
                .build();

        QueryRequest queryRequest = QueryRequest
                .newBuilder()
                .addQueries(queryVector)
                .setTopK(maxResults)
                .build();

        List<String> matchedVectorIds = connection.getBlockingStub()
                .query(queryRequest)
                .getResultsList()
                .get(0) // TODO ?
                .getMatchesList()
                .stream()
                .map(ScoredVector::getId)
                .collect(toList());

        Collection<Vector> matchedVectors = connection.getBlockingStub().fetch(FetchRequest.newBuilder()
                        .addAllIds(matchedVectorIds)
                        .setNamespace(nameSpace)
                        .build())
                .getVectorsMap()
                .values();

        return matchedVectors.stream()
                .map(PineconeDatabase::toEmbedding)
                .collect(toList());
    }

    private static Embedding toEmbedding(Vector vector) {
        String text = vector.getMetadata()
                .getFieldsMap()
                .get(METADATA_ORIGINAL_TEXT)
                .getStringValue();

        return new Embedding(text, toDoubles(vector.getValuesList()));
    }

    private static List<Float> toFloats(Embedding embedding) {
        return embedding.vector().stream()
                .map(Double::floatValue)
                .collect(toList());
    }

    private static List<Double> toDoubles(List<Float> floats) {
        return floats.stream()
                .map(Float::doubleValue)
                .collect(toList());
    }

    private static String createUniqueId() {
        return UUID.randomUUID().toString();
    }
}
