import pandas as pd
from pathlib import Path


class SHLDataEnricher:
    def __init__(self):
        # Official SHL test type meanings
        self.test_type_map = {
            "A": "Ability & Aptitude",
            "B": "Biodata & Situational Judgement",
            "C": "Competencies",
            "D": "Development & 360",
            "E": "Assessment Exercises",
            "K": "Knowledge & Skills",
            "P": "Personality & Behavior",
            "S": "Simulations"
        }

        self.test_type_descriptions = {
            "A": "Cognitive abilities such as numerical, verbal, and logical reasoning",
            "B": "Past behavior and situational decision-making",
            "C": "Role-specific competencies and skills",
            "D": "Developmental feedback and growth assessment",
            "E": "Work-sample based assessment exercises",
            "K": "Technical and functional knowledge assessment",
            "P": "Personality traits, work style, and behavior",
            "S": "Realistic simulations of job tasks"
        }

    def enrich(self):
        print("🔄 Starting data enrichment...")

        input_path = Path("backend/data/shl_individual_tests.csv")
        output_path = Path("backend/data/shl_products_enriched.csv")

        df = pd.read_csv(input_path)

        # Ensure test_types_list exists
        df["test_types_list"] = df["test_types_list"].apply(
            lambda x: eval(x) if isinstance(x, str) else []
        )

        # Map full test type names
        df["test_types_full"] = df["test_types_list"].apply(
            lambda types: [self.test_type_map.get(t, t) for t in types]
        )

        # Infer roles
        df["inferred_roles"] = df["name"].apply(self._infer_roles)

        # Create RAG-friendly description
        df["embedding_text"] = df.apply(self._build_embedding_text, axis=1)

        # Save enriched dataset
        df.to_csv(output_path, index=False, encoding="utf-8")

        print(f"✅ Enriched data saved at: {output_path}")
        print(f"📊 Total enriched assessments: {len(df)}")

        return df

    def _build_embedding_text(self, row):
        parts = [
            f"Assessment Name: {row['name']}",
            f"Assessment Type: {', '.join(row['test_types_full'])}",
        ]

        if row.get("remote_testing"):
            parts.append("Remote testing available")

        if row.get("adaptive_irt"):
            parts.append("Adaptive testing supported")

        # Add test type meanings
        for t in row["test_types_list"]:
            if t in self.test_type_descriptions:
                parts.append(
                    f"{self.test_type_map[t]} assessment: {self.test_type_descriptions[t]}"
                )

        # Add inferred roles
        parts.append(f"Suitable for roles such as {', '.join(row['inferred_roles'])}")

        return ". ".join(parts)

    def _infer_roles(self, name):
        name = name.lower()
        roles = []

        mapping = {
            "java": ["Java Developer", "Backend Engineer"],
            "python": ["Python Developer", "Software Engineer"],
            "developer": ["Software Developer"],
            "analyst": ["Analyst", "Business Analyst"],
            "manager": ["Manager", "Team Lead"],
            "sales": ["Sales Professional"],
            "customer": ["Customer Support"],
            "engineer": ["Engineer"],
            "graduate": ["Graduate", "Entry Level"]
        }

        for key, values in mapping.items():
            if key in name:
                roles.extend(values)

        return roles if roles else ["General"]

def main():
    print("=" * 70)
    print("SHL DATA ENRICHMENT (INDIVIDUAL TESTS ONLY)")
    print("=" * 70)

    enricher = SHLDataEnricher()
    enricher.enrich()

    print("\n🎯 Next step: Embeddings + FAISS indexing")
    print("=" * 70)


if __name__ == "__main__":
    main()
