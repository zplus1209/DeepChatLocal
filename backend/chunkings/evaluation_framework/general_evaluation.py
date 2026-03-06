from pathlib import Path
from importlib import resources

from backend.chunkings.evaluation_framework.base_evaluation import BaseChunkingEvaluation


class GeneralChunkingEvaluation(BaseChunkingEvaluation):
    def __init__(self, chroma_db_path=None):
        with resources.as_file(resources.files('backend.chunkings.evaluation_framework') / 'general_evaluation_data') as general_benchmark_path:
            self.general_benchmark_path = general_benchmark_path
            questions_df_path = self.general_benchmark_path / 'questions_df.csv'

            corpora_folder_path = self.general_benchmark_path / 'corpora'
            corpora_filenames = [f for f in corpora_folder_path.iterdir() if f.is_file()]

            corpora_id_paths = {
                f.stem: str(f) for f in corpora_filenames
            }

            super().__init__(str(questions_df_path), chroma_db_path=chroma_db_path, corpora_id_paths=corpora_id_paths)

if __name__ == "__main__":
    from backend.chunkings import (
        RecursiveTokenChunker,
        ClusterSemanticChunker,
        LLMSemanticChunker
    )
    from backend.llms import EmbeddingModel
    import json
    
    evaluation = GeneralChunkingEvaluation()

    embedding_model = EmbeddingModel(engine="hf", model_name="sentence-transformers/all-MiniLM-L6-v2")

    chunker = LLMSemanticChunker()

    results = evaluation.run(chunker, embedding_model)
    with open("./result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)