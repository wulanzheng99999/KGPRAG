"""
图存储模块：Neo4j 图数据库操作
"""
from __future__ import annotations

from typing import List, Dict, Set

from neo4j import GraphDatabase

from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from util.text_utils import normalize_entity


class GraphStore:
    """Neo4j 图存储管理器"""
    
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
    
    def reset(self):
        """清空图数据库并创建索引"""
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                session.run("CREATE INDEX chunk_id_idx IF NOT EXISTS FOR (c:Chunk) ON (c.id)")
                session.run("CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)")
        except Exception as e:
            print(f"⚠️ Neo4j Reset Error: {e}")

    def get_existing_chunk_ids(self) -> Set[str]:
        """获取所有已存在的 Chunk ID"""
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (c:Chunk) RETURN c.id AS id")
                return {record["id"] for record in result}
        except Exception as e:
            print(f"⚠️ Get Existing Chunk IDs Error: {e}")
            return set()
    
    def write_chunks(self, chunks: List[Dict]):
        """写入 Chunks、Entities 和关系"""
        BATCH_SIZE = 2000
        total = len(chunks)
        print(f"[Neo4j] Writing {total} chunks in batches of {BATCH_SIZE}...")
        
        try:
            with self.driver.session() as session:
                for i in range(0, total, BATCH_SIZE):
                    batch = chunks[i : i + BATCH_SIZE]
                    
                    # 写入 Chunks 和 Entities
                    session.run(
                        """
                        UNWIND $batch AS row
                        MERGE (d:Document {title: row.doc_title})
                        MERGE (c:Chunk {id: row.chunk_id}) SET c.text = row.text
                        MERGE (d)-[:CONTAINS]->(c)
                        
                        FOREACH (_ IN CASE WHEN row.prev_id IS NOT NULL THEN [1] ELSE [] END |
                            MERGE (p:Chunk {id: row.prev_id}) MERGE (p)-[:NEXT]->(c))
                        
                        FOREACH (e IN row.entities |
                            MERGE (ent:Entity {name: e.name}) SET ent.type = e.type
                            MERGE (c)-[:MENTIONS]->(ent))
                        """,
                        batch=batch,
                    )
                    
                    # 写入 REBEL 抽取的关系三元组
                    session.run(
                        """
                        UNWIND $batch AS row
                        UNWIND row.rebel_rels AS rel
                        MERGE (source:Entity {name: rel.source})
                        MERGE (target:Entity {name: rel.target})
                        MERGE (source)-[r:RELATION {type: rel.type, source_chunk: row.chunk_id}]->(target)
                        """,
                        batch=batch
                    )
                    print(f"[Neo4j] Chunks Progress: {min(i + BATCH_SIZE, total)}/{total}")
        except Exception as e:
            print(f"❌ Neo4j Write Chunks Error: {e}")
    
    def write_summaries(self, summary_nodes: List[Dict], summary_rels: List[Dict]):
        """写入摘要节点和关系"""
        if not summary_nodes:
            return
            
        BATCH_SIZE = 2000
        
        try:
            with self.driver.session() as session:
                # 1. 批量写入节点
                total_nodes = len(summary_nodes)
                print(f"[Neo4j] Writing {total_nodes} summaries...")
                for i in range(0, total_nodes, BATCH_SIZE):
                    batch = summary_nodes[i : i + BATCH_SIZE]
                    session.run(
                        """
                        UNWIND $batch AS row
                        MERGE (s:Summary {id: row.id}) SET s.text = row.text, s.level = row.level
                        MERGE (d:Document {title: row.doc_title}) MERGE (d)-[:HAS_SUMMARY]->(s)
                        """,
                        batch=batch,
                    )
                    print(f"[Neo4j] Summaries Progress: {min(i + BATCH_SIZE, total_nodes)}/{total_nodes}")
                
                # 分离关系类型
                rels_to_summary = [r for r in summary_rels if r["type"] == "CONTAINS_SUMMARY"]
                rels_to_chunk = [r for r in summary_rels if r["type"] == "CONTAINS_CHUNK"]

                # 2. 批量写入 Summary-Summary 关系
                if rels_to_summary:
                    total_rs = len(rels_to_summary)
                    print(f"[Neo4j] Writing {total_rs} summary-summary edges...")
                    for i in range(0, total_rs, BATCH_SIZE):
                        batch = rels_to_summary[i : i + BATCH_SIZE]
                        session.run(
                            """
                            UNWIND $batch AS row
                            MATCH (source:Summary {id: row.source}), (target:Summary {id: row.target})
                            MERGE (source)-[:CONTAINS]->(target)
                            """,
                            batch=batch
                        )
                        print(f"[Neo4j] Summary-Summary Edges Progress: {min(i + BATCH_SIZE, total_rs)}/{total_rs}")

                # 3. 批量写入 Summary-Chunk 关系
                if rels_to_chunk:
                    total_rc = len(rels_to_chunk)
                    print(f"[Neo4j] Writing {total_rc} summary-chunk edges...")
                    for i in range(0, total_rc, BATCH_SIZE):
                        batch = rels_to_chunk[i : i + BATCH_SIZE]
                        session.run(
                            """
                            UNWIND $batch AS row
                            MATCH (source:Summary {id: row.source}), (target:Chunk {id: row.target})
                            MERGE (source)-[:CONTAINS]->(target)
                            """,
                            batch=batch
                        )
                        print(f"[Neo4j] Summary-Chunk Edges Progress: {min(i + BATCH_SIZE, total_rc)}/{total_rc}")
        except Exception as e:
            print(f"❌ Neo4j Write Summaries Error: {e}")
    
    def write_semantic_edges(self, semantic_rels: List[Dict]):
        """写入语义相似边（:RELATED 关系，基于 embedding 相似度）"""
        if not semantic_rels:
            return
        
        BATCH_SIZE = 5000
        total = len(semantic_rels)
        print(f"[Neo4j] Writing {total} semantic edges...")
            
        try:
            with self.driver.session() as session:
                for i in range(0, total, BATCH_SIZE):
                    batch = semantic_rels[i : i + BATCH_SIZE]
                    session.run(
                        """
                        UNWIND $batch AS row
                        MATCH (a:Chunk {id: row.source}), (b:Chunk {id: row.target})
                        MERGE (a)-[r:RELATED]-(b) SET r.score = row.score
                        """,
                        batch=batch,
                    )
                    print(f"[Neo4j] Semantic Edges Progress: {min(i + BATCH_SIZE, total)}/{total}")
        except Exception as e:
            print(f"❌ Neo4j Write Semantic Edges Error: {e}")
    
    def write_entity_bridge_edges(self, bridge_rels: List[Dict]):
        """
        写入实体共现硬边（:ENTITY_BRIDGE 关系，基于共享稀有实体）
        与 :RELATED 分离，避免覆盖语义相似度分数
        """
        if not bridge_rels:
            return
            
        BATCH_SIZE = 5000
        total = len(bridge_rels)
        print(f"[Neo4j] Writing {total} entity bridge edges...")

        try:
            with self.driver.session() as session:
                for i in range(0, total, BATCH_SIZE):
                    batch = bridge_rels[i : i + BATCH_SIZE]
                    session.run(
                        """
                        UNWIND $batch AS row
                        MATCH (a:Chunk {id: row.source}), (b:Chunk {id: row.target})
                        MERGE (a)-[r:ENTITY_BRIDGE]-(b) SET r.score = row.score
                        """,
                        batch=batch,
                    )
                    print(f"[Neo4j] Bridge Edges Progress: {min(i + BATCH_SIZE, total)}/{total}")
        except Exception as e:
            print(f"❌ Neo4j Write Entity Bridge Edges Error: {e}")
    
    def expand_node(self, chunk_id: str, visited: Set[str], query_entities: List[str] = None) -> Dict[str, Dict]:
        """
        扩展节点的邻居
        包含：Sequential + Semantic + Entity Bridge (硬边) + Entity Co-mention + REBEL Relation Path
        """
        data: Dict[str, Dict] = {}
        query_entities = query_entities or []
        
        # 基础扩展查询（不含硬边，避免混淆）
        base_query = """
        MATCH (s:Chunk {id: $id})
        // 1. Sequential expansion
        OPTIONAL MATCH (s)-[:NEXT]-(n:Chunk) WHERE NOT n.id IN $vis
        // 2. Semantic similarity expansion (仅 :RELATED 边，阈值过滤)
        OPTIONAL MATCH (s)-[r:RELATED]-(sem:Chunk) WHERE r.score > 0.7 AND NOT sem.id IN $vis
        // 3. Entity co-mention expansion (通过 :MENTIONS 边)
        OPTIONAL MATCH (s)-[:MENTIONS]->(:Entity)<-[:MENTIONS]-(b:Chunk) WHERE NOT b.id IN $vis
        // 4. REBEL Relation Path expansion
        OPTIONAL MATCH (s)-[:MENTIONS]->(:Entity)-[:RELATION]->(:Entity)<-[:MENTIONS]-(rel:Chunk) WHERE NOT rel.id IN $vis
        
        RETURN n.id, n.text, sem.id, sem.text, b.id, b.text, rel.id, rel.text LIMIT 15
        """
        
        try:
            with self.driver.session() as s:
                res = s.run(base_query, id=chunk_id, vis=list(visited))
                for r in res:
                    if r["n.id"]:
                        data[r["n.id"]] = {"text": r["n.text"], "title": "Seq"}
                    if r["sem.id"]:
                        data[r["sem.id"]] = {"text": r["sem.text"], "title": "Sem"}
                    if r["b.id"]:
                        data[r["b.id"]] = {"text": r["b.text"], "title": "EntMention"}
                    if r["rel.id"]:
                        data[r["rel.id"]] = {"text": r["rel.text"], "title": "RelPath"}
        except Exception as e:
            print(f"⚠️ Graph Expand Error: {e}")
        
        # 5. Entity Bridge 硬边扩展（独立查询，独立配额）
        # 硬边是基于稀有实体共现预计算的，优先级高于普通 co-mention
        hard_edge_query = """
        MATCH (s:Chunk {id: $id})-[:ENTITY_BRIDGE]-(bridge:Chunk)
        WHERE NOT bridge.id IN $vis
        RETURN bridge.id AS id, bridge.text AS text LIMIT 10
        """
        try:
            with self.driver.session() as s:
                res = s.run(hard_edge_query, id=chunk_id, vis=list(visited))
                for r in res:
                    if r["id"] and r["id"] not in data:
                        data[r["id"]] = {"text": r["text"], "title": "EntBridge"}
        except Exception as e:
            print(f"⚠️ Hard Edge Expand Error: {e}")
        
        # [思路E] Query-Guided Entity Linking with Fuzzy Matching
        if query_entities:
            for qe in query_entities:
                qe_norm = normalize_entity(qe)
                if not qe_norm:
                    continue
                    
                entity_query = """
                MATCH (ent:Entity)<-[:MENTIONS]-(c:Chunk)
                WHERE (ent.name CONTAINS $entity OR $entity CONTAINS ent.name) 
                      AND NOT c.id IN $vis
                RETURN DISTINCT c.id AS id, c.text AS text LIMIT 5
                """
                try:
                    with self.driver.session() as s:
                        res = s.run(entity_query, entity=qe_norm, vis=list(visited))
                        for r in res:
                            if r["id"] and r["id"] not in data:
                                data[r["id"]] = {"text": r["text"], "title": "QueryEnt"}
                except Exception:
                    pass
        
        return data
    
    def get_summary_children(self, summary_id: str) -> List[Dict]:
        """获取摘要节点的子节点"""
        children = []
        try:
            with self.driver.session() as s:
                res = s.run(
                    """
                    MATCH (parent:Summary {id: $id})-[:CONTAINS]->(child)
                    RETURN child.id AS id, child.text AS text, labels(child)[0] AS type
                    """,
                    id=summary_id,
                )
                for r in res:
                    node_type = r["type"]
                    children.append({
                        "id": r["id"], 
                        "text": r["text"], 
                        "type": node_type
                    })
        except Exception as e:
            print(f"⚠️ Get Summary Children Error: {e}")
        return children
    
    def close(self):
        if self.driver:
            self.driver.close()
