import os
import sys
import json
import logging
from neo4j import GraphDatabase

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the parent directory is in the system path to find the config module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

class Neo4jSeeder:
    """
    Seeds the Neo4j database from a JSON configuration file.
    This script is designed to be idempotent, so it can be run multiple times safely.
    """

    def __init__(self, uri, user, password, json_path):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.json_path = json_path
        self.data = self._load_json_data()

    def _load_json_data(self):
        """Loads the patterns from the JSON file."""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"❌ Configuration file not found at {self.json_path}")
            sys.exit(1)
        except json.JSONDecodeError:
            logging.error(f"❌ Failed to decode JSON from {self.json_path}")
            sys.exit(1)

    def close(self):
        """Closes the database connection."""
        self.driver.close()

    def seed_database(self):
        """Runs the entire seeding process."""
        if not self.data:
            return

        logging.info("🚀 Starting Neo4j database seeding process...")
        self._create_constraints()
        self._seed_entities()
        self._seed_aliases()
        self._seed_subfaction_relations()
        logging.info("✅ Database seeding completed successfully!")

    def _create_constraints(self):
        """Create uniqueness constraints for each entity type to optimize MERGE and prevent duplicates."""
        with self.driver.session() as session:
            entity_types = list(self.data.get("entity_patterns", {}).keys())
            for entity_type in entity_types:
                label = entity_type.capitalize()
                constraint_name = f"constraint_unique_{label}_name"
                query = f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS FOR (n:{label}) REQUIRE n.name IS UNIQUE"
                session.run(query)
                logging.info(f"Ensured constraint on :{label}(name) exists.")

    def _seed_entities(self):
        """Seeds all entities from the 'entity_patterns' section."""
        entity_patterns = self.data.get("entity_patterns", {})
        with self.driver.session() as session:
            for entity_type, entities in entity_patterns.items():
                label = entity_type.capitalize()
                for entity_name in entities:
                    query = f"MERGE (n:{label} {{name: $name}})"
                    session.run(query, name=entity_name)
            logging.info(f"Seeded {sum(len(v) for v in entity_patterns.values())} entities across {len(entity_patterns)} types.")

    def _seed_aliases(self):
        """Seeds alias relationships from the 'alias_mapping' section."""
        alias_mapping = self.data.get("alias_mapping", {})
        if not alias_mapping:
            logging.info("No alias mappings to seed.")
            return
            
        with self.driver.session() as session:
            count = 0
            for entity_name, aliases in alias_mapping.items():
                for alias in aliases:
                    # First, ensure the alias exists as a node. We don't know its type, so we create it with a generic 'Entity' label if it doesn't exist.
                    session.run("MERGE (a:Entity {name: $alias_name})", alias_name=alias)
                    
                    # Now, create the relationship
                    query = """
                    MATCH (entity {name: $entity_name})
                    MATCH (alias {name: $alias_name})
                    MERGE (alias)-[:ALIAS_OF]->(entity)
                    """
                    session.run(query, entity_name=entity_name, alias_name=alias)
                    count += 1
            logging.info(f"Seeded {count} alias relationships.")

    def _seed_subfaction_relations(self):
        """Seeds 'IS_SUBFACTION_OF' relationships."""
        subfaction_relations = self.data.get("subfaction_relations", {})
        if not subfaction_relations:
            logging.info("No sub-faction relationships to seed.")
            return

        with self.driver.session() as session:
            count = 0
            for subfaction_name, faction_name in subfaction_relations.items():
                query = """
                MATCH (sub:Sub_faction {name: $subfaction_name})
                MATCH (faction:Faction {name: $faction_name})
                MERGE (sub)-[:IS_SUBFACTION_OF]->(faction)
                """
                session.run(query, subfaction_name=subfaction_name, faction_name=faction_name)
                count += 1
            logging.info(f"Seeded {count} sub-faction relationships.")

if __name__ == "__main__":
    # The script expects the patterns file to be in the same directory.
    patterns_file_path = os.path.join(os.path.dirname(__file__), 'knowledge_graph_patterns.json')
    
    seeder = Neo4jSeeder(
        uri=NEO4J_URI,
        user=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        json_path=patterns_file_path
    )
    
    seeder.seed_database()
    seeder.close() 