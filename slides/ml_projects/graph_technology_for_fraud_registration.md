# **Technical Analysis of Graph-Based Systems for Registration Fraud Mitigation**

## **Legacy Relational Vulnerabilities in Mitigating Registration Fraud**

Digital platforms, financial networks, and e-commerce ecosystems face a continuous, highly adaptive threat from registration fraud.1 Unlike traditional transactional fraud—where a compromise is typically restricted to an isolated payment event—registration fraud is fundamentally systemic.3 It involves the creation of synthetic identities, constructed by combining stolen, manipulated, and completely fabricated attributes such as physical addresses, phone numbers, email handles, and device fingerprints.2 Once registered, these accounts act as sleep cells, coordinating in large-scale rings to orchestrate promotional abuse, execute account takeovers, distribute money laundering networks, or launch coordinated transactional fraud schemes.3  
Legacy fraud prevention frameworks are structurally and computationally ill-equipped to counter these operations.1 Traditional architectures are primarily event-driven and rule-based.1 They evaluate transaction or registration parameters in isolation, checking static parameters such as transaction amounts, or testing whether a registrant's IP address belongs to a high-risk region.1 Because each synthetic profile is crafted to mimic the behavior of a legitimate user, these isolated profiles easily bypass rule-based boundary checks.1  
From a database perspective, relational database management systems (RDBMS) fail to scale for relation-heavy detection tasks.1 Relational architectures store data in normalized, isolated tables.5 Resolving connections between entities—such as identifying whether multiple newly registered accounts share a physical address, a phone number, or an underlying hardware device—requires dynamic SQL table joins.2  
As the depth of an investigation extends beyond two hops, the required self-joins and recursive operations grow exponentially in complexity.8 This structural limitation causes severe geometric performance degradation, leading to query latencies and CPU starvation.1 This makes real-time, preventative intervention during account creation computationally impossible.1  
To illustrate this structural challenge, a typical relational schema utilized during user registration is structured across several tables within a database, requiring complex keys to link related entities.10

| Relational Table Name | Core Data Fields and Schema | Cardinality and Relationships |
| :---- | :---- | :---- |
| customers | customer\_id, name, creation\_timestamp, risk\_label.10 | Primary entity representing registered users.10 |
| emails | email\_id, email\_address, domain\_age, is\_disposable.10 | Distinct email accounts registered on the platform.10 |
| phones | phone\_id, phone\_number, carrier\_type, country\_code.10 | Telephone numbers associated with customer profiles.10 |
| addresses | address\_id, normalized\_street, city, zip\_code.10 | Physical billing and shipping locations.10 |
| customer\_emails | customer\_id, email\_id.10 | Many-to-many join table for profiles and email addresses.10 |
| customer\_phones | customer\_id, phone\_id.10 | Many-to-many join table for profiles and phone numbers.10 |
| customer\_addresses | customer\_id, address\_id.10 | Many-to-many join table for profiles and geographic locations.10 |
| orders | order\_id, customer\_id, amount, is\_fraud\_flagged.10 | Transaction records linked back to registered profiles.10 |

## **Unified Graph Modeling and Schema Definitions for Identity Verification**

Property graphs address the limitations of relational models by treating data connections as first-class physical entities.7 In a property graph model, entities are represented as nodes, and their direct relationships are modeled as edges.2 This shift allows security platforms to model registration attempts as a connected network, exposing the topological signatures of fraud rings.2

### **Identity Graph Node and Edge Typology**

To construct a graph for identity verification, a platform defines a heterogeneous network schema where personal identifiers and hardware footprints are mapped alongside customer profiles.2

| Node Label | Node Description | Key Properties and Constraints |
| :---- | :---- | :---- |
| Customer | Represents the unique user registration account.10 | customer\_id (Primary Key), creation\_time, fraud\_risk.10 |
| Email | Represents email addresses used during account creation.10 | email\_id (Primary Key), normalized\_handle, is\_disposable.10 |
| Phone | Represents phone numbers provided by the registrants.10 | phone\_id (Primary Key), phone\_number, carrier\_type.10 |
| Address | Represents physical billing or shipping addresses.10 | address\_id (Primary Key), normalized\_address, coordinates.10 |
| Device | Represents hardware profiles and unique fingerprints.8 | device\_id (Primary Key), operating\_system, biometric\_hash.3 |
| IP | Represents the network address initiating the registration.2 | ip\_address (Primary Key), asn\_number, is\_proxy.2 |

| Edge Label | Source Node | Target Node | Edge Description and Context |
| :---- | :---- | :---- | :---- |
| HasEmail | Customer | Email | Links a profile to their registered email address.10 |
| HasPhone | Customer | Phone | Links a profile to their registered phone number.10 |
| HasAddress | Customer | Address | Maps the physical location associated with the profile.10 |
| RegisteredVia | Customer | Device | Maps the physical terminal used to execute the registration.8 |
| OriginatedFrom | Customer | IP | Tracks the network origin of the registration attempt.2 |

### **BigQuery Graph Schema Construction and DDL**

To implement a property graph, data architects must hold the appropriate administrative privileges.14 In Google Cloud, administrators must grant the BigQuery Data Editor IAM role (roles/bigquery.dataEditor) on the specific dataset where the underlying node tables, edge tables, and logical graphs are compiled.14  
BigQuery Graph stores graphs as logical views over existing tables in Colossus, Google’s distributed file system.14 This architecture eliminates data duplication and the need for complex ETL pipelines, as the property graph acts as an analytical layer over standard relational tables.8  
Consequently, organizations are charged only once for data storage based on standard BigQuery storage pricing, regardless of how many logical graphs are defined over the tables.5  
Using the Data Definition Language (DDL) supported by the ISO Property Graph Queries (SQL/PGQ) standard, a unified identity graph is defined over base tables in BigQuery as follows 5:

SQL  
CREATE OR REPLACE PROPERTY GRAPH fraud\_demo.IdentityGraph  
NODE TABLES (  
  fraud\_demo.customers KEY (customer\_id),  
  fraud\_demo.emails KEY (email\_id),  
  fraud\_demo.phones KEY (phone\_id),  
  fraud\_demo.addresses KEY (address\_id)  
)  
EDGE TABLES (  
  fraud\_demo.customer\_emails AS HasEmail  
    KEY (customer\_id, email\_id)  
    SOURCE KEY (customer\_id) REFERENCES customers  
    DESTINATION KEY (email\_id) REFERENCES emails,  
  fraud\_demo.customer\_phones AS HasPhone  
    KEY (customer\_id, phone\_id)  
    SOURCE KEY (customer\_id) REFERENCES customers  
    DESTINATION KEY (phone\_id) REFERENCES phones,  
  fraud\_demo.customer\_addresses AS HasAddress  
    KEY (customer\_id, address\_id)  
    SOURCE KEY (customer\_id) REFERENCES customers  
    DESTINATION KEY (address\_id) REFERENCES addresses  
);

This definition establishes a queryable logical view, allowing analysts to run Graph Query Language (GQL) operations over standard physical tables.5

## **Algorithmic Resolution and Topological Analysis of Coordinated Fraud Rings**

Coordinated fraud rings are characterized by specific patterns of shared resources, automated behavior, and abnormal network connectivity.3 To detect these structures at scale, platforms combine entity resolution with graph partitioning algorithms.7

### **Entity Resolution Rules**

Fraudsters attempt to evade detection by introducing slight variations into their registration credentials.3 To address this, platforms implement database-native full-text search with fuzzy matching and address normalization.3  
Once these variations are resolved, the system applies programmatic business rules to link separate customer profiles.13 For example, the **Shared Identifiers Rule** establishes that if two distinct customer accounts share a highly specific identifier (such as a device fingerprint or physical address connected to ten or fewer total accounts), and also share at least two other secondary identifiers (such as an IP address or a credit card), they are programmatically linked via a new, explicit edge labeled SHARED\_IDS.13

\[Customer Node A\] \-- (HasAddress) \--\> \[Address Node: Normalized\] \<-- (HasAddress) \--  
       |                                                                                    |  
 (RegisteredVia)                                                                      (RegisteredVia)  
       |                                                                                    |  
       v                                                                                    v  
 \<--------------------------------------------------------------------  
         
       \=================== RESOLVED VIA SYSTEM BUSINESS RULE \===================  
         
\[Customer Node A\] \<---------------------- (SHARED\_IDS) \-----------------------------\>

### **Community Detection and Resource Partitioning**

After establishing resolved relationships, community detection algorithms partition the graph to isolate fraud rings.4

* **Weakly Connected Components (WCC)**: WCC groups nodes connected by any path, regardless of edge direction.7 WCC acts as a deterministic filter, grouping users into disjoint communities based on shared identifiers.7 If a single profile within a WCC community is flagged as fraudulent, the entire cluster is compromised.13 The system dynamically labels all accounts within that community as FraudRiskUser (fraudRisk=1), exposing connected accounts that bypassed traditional checks.13  
* **Strongly Connected Components (SCC)**: SCC isolates tightly bound, directed loops where every node can reach every other node in both directions.7 In registration fraud, SCC is utilized to detect transactional loops, automated circular referrals, or money-laundering pipelines (where Account A transfers funds to Account B, which routes to Account C, and eventually loops back to Account A).3  
* **Louvain Modularity Clustering**: While connected components show if a path exists, Louvain measures the tightness of a group by optimizing the graph’s modularity.4 Louvain partitions the network into dense communities that interact far more with each other than with the rest of the graph.4 This allows investigators to differentiate between massive, benign communities (such as users sharing a public university IP network) and highly concentrated fraud rings operating within that network.4

### **Topological Anomalies**

Coordinated fraud rings exhibit specific topological signatures that differ sharply from legitimate user networks.3

* **The Fan-Out Anomaly**: This occurs when a single identifier node (such as a unique Device fingerprint or a single physical Address) is connected to an anomalously high number of distinct Customer nodes.3 In a legitimate context, a device is shared by only a few family members; a fan-out of fifty accounts registered to one device within forty-eight hours is a strong indicator of automated script registration or virtual machine manipulation.3  
* **The Fan-In Anomaly**: This signature involves multiple newly registered Customer nodes rapidly funneling assets, referrals, or promotional codes to a single target entity.3 This is common in registration-based promotion abuse, where hundreds of fake accounts are created to generate referral bonuses for a master account.3  
* **Network Size Differential (The diff Metric)**: By combining graph analytics with transactional datasets, platforms track the expansion rate of connected networks over defined observation windows.10 The network size differential (diff) calculates the change in network size between the "before" and "after" states of an account's connected subgraph.10 A rapid, abnormal increase in the diff metric indicates an active, resource-sharing fraud ring expanding across the platform.10

To evaluate risk programmatically, systems calculate graph-theoretical metrics across the network.2 One key metric is the Local Clustering Coefficient (![][image1]), which measures the connectivity density among the neighbors of a given node ![][image2].12 Mathematically, it is expressed as:  
![][image3]  
In this equation, ![][image4] represents the number of active edges connecting the neighbors of node ![][image2], and ![][image5] is the degree (total number of connected edges) of node ![][image2].12 Legitimate users typically exhibit low local clustering coefficients on their shared resource nodes, as their devices and IP addresses do not overlap in highly dense, closed-loop configurations.12 Conversely, a highly dense, fully connected subgraph of users, devices, and addresses yields a local clustering coefficient close to ![][image6], indicating a highly collusive, automated registration ring.12

## **Computational Complexity and Expressiveness: Graph Query Language versus Relational SQL**

The primary advantage of graph technology over relational SQL lies in its computational efficiency and code expressiveness when traversing multi-hop networks.5  
In a traditional relational database, executing a multi-hop traversal requires a series of self-joins.5 For example, tracing a 4-hop path across standard tables involves joining the tables repeatedly on primary and foreign keys.5 Each join requires the database to scan indexes or perform hash-joins across the entire dataset, creating an ![][image7] computational footprint (where ![][image8] is the table size and ![][image9] is the depth of the hop traversal).7 Consequently, even a simple 3-hop query on a moderately sized dataset can easily stall a relational engine.7  
Graph databases resolve this through index-free adjacency.7 In a native graph database, relationships are stored as physical, direct memory pointers on disk and in RAM.7 Traversing an edge does not require a global index lookup; the engine simply reads the pointers stored with the node, hopping directly to the target memory address.7 The computational complexity of a traversal is reduced to ![][image10], where ![][image11] represents the average node degree (typically a small constant) and ![][image9] is the traversal depth.7 This transition from global index scans to localized pointer dereferencing maintains fast, sub-millisecond query response times even as the total database size scales to billions of records.3  
This performance difference is accompanied by a major shift in code expressiveness.5 Writing deep traversal logic in SQL requires verbose, nested, and fragile recursive Common Table Expressions (CTEs).5 In contrast, Graph Query Language (GQL) and Cypher allow developers to declaratively "draw" the target relationship pattern using intuitive ascii-art syntax (parentheses () for nodes, brackets \`\` for edges, and arrows \-\> or \<- for direction).5 Filters are specified inline using braces {}.18  
GQL execution operates on the concept of a "working table".20 Each GQL query consists of one or more linear query statements.19 When a statement executes, it receives an incoming working table, performs graph pattern matching or filtering, and outputs an outgoing working table containing intermediate results, which is then passed to the next statement in the chain.20 The first incoming working table is a single-row table, and the final outgoing table is returned as the final query result.20  
Statements are chained together using the NEXT keyword, and multiple linear query statements can be combined using set operators such as UNION ALL.19 Additionally, GQL supports path patterns with quantifiers (such as {1, 3}) to find paths that repeat a specific edge pattern within a defined range, making multi-hop path traversals highly expressive.19  
To illustrate these syntax differences, consider a common fraud detection query that identifies accounts linked to a blocked customer profile 7:

### **The GQL / Cypher Implementation**

Cypher  
GRAPH graph\_db.IdentityGraph  
MATCH (blocked:Customer {is\_blocked: TRUE})--\>{1, 3}(target:Customer)  
RETURN target.customer\_id AS suspicious\_account

### **The Relational SQL Implementation (Recursive CTE)**

SQL  
WITH RECURSIVE undirected AS (  
  SELECT src AS a, dst AS b FROM edges  
  UNION  
  SELECT dst AS a, src AS b FROM edges  
), nodes AS (  
  SELECT a AS node FROM undirected  
  UNION  
  SELECT b FROM undirected  
),   
reach(node, root, depth) AS (  
  SELECT customer\_id, customer\_id, 0   
  FROM customers   
  WHERE is\_blocked \= TRUE  
  UNION ALL  
  SELECT u.b, r.root, r.depth \+ 1  
  FROM reach r  
  JOIN undirected u ON u.a \= r.node  
  WHERE r.depth \< 3  
),   
components AS (  
  SELECT DISTINCT node   
  FROM reach   
  WHERE node\!= root  
)  
SELECT node AS suspicious\_account FROM components;

This comparison highlights the operational differences between the two paradigms.7 The SQL recursive join is difficult to maintain and tune.5 Furthermore, because it executes massive, global table joins at every iteration, it consumes substantial computational resources, making it impractical for large datasets.7 The GQL query runs within a dedicated graph engine, completing the operation in a fraction of the time with minimal memory overhead.7

## **Next-Generation Hybrid Architectures: Spanner Graph and BigQuery Graph Integration**

To protect digital platforms from registration fraud, enterprise architectures must support both real-time preventative intervention and comprehensive historical analysis.16 Google Cloud addresses these requirements by integrating Spanner Graph and BigQuery Graph into a unified, multi-tiered architecture.8  
This framework operates over a single, consistent data schema and utilizes the standardized ISO GQL query language, eliminating the need to move or duplicate data.8

                \+---------------------------------------+  
                |          RELATIONAL STORAGE           |  
                |  (Spanner / BigQuery Base Tables)    |  
                \+-------------------+-------------------+  
                                    |  
                    \+---------------+---------------+  
                    |                               |  
                    v                               v  
         \+--------------------+          \+--------------------+  
         |   SPANNER GRAPH    |          |   BIGQUERY GRAPH   |  
         | (Operational/OLTP) |          | (Analytical/OLAP)  |  
         |                    |          |                    |  
         | \- Sub-10ms Latency |          | \- Billions of Nodes|  
         | \- Real-Time Block  |          | \- Global ML/WCC    |  
         \+--------------------+          \+--------------------+

### **Operational vs. Analytical Graph Engines**

The platform deploys Spanner Graph for operational, real-time protection and BigQuery Graph for global, historical analysis 16:

* **Spanner Graph (Operational Layer / OLTP)**: Optimized for high-throughput, low-latency transaction processing.16 Spanner Graph executes operational graph queries within milliseconds during the actual registration or checkout process.16 It performs fast 1-to-3 hop lookups starting from a single node, checking if the current registrant is directly linked to a known blocked device, physical address, or IP range.16  
* **BigQuery Graph (Analytical Layer / OLAP)**: Optimized for running complex, resource-intensive queries and graph algorithms over massive, historical datasets.16 It scales to billions of nodes and edges.8 Instead of evaluating a single registration attempt, BigQuery Graph runs global community detection algorithms (such as Louvain and WCC), matches multi-hop patterns spanning 4 to 7+ hops, and builds large-scale knowledge graphs.8

| Architectural Dimension | Spanner Graph (OLTP) | BigQuery Graph (OLAP) |
| :---- | :---- | :---- |
| **Primary Workload** | Real-time transaction validation.16 | Historical analysis and ML training.16 |
| **Typical Query Latency** | Sub-milliseconds to milliseconds.21 | Seconds to hours.21 |
| **Data Ingestion** | Live application streams.21 | Historical data lakes and archives.21 |
| **Compute Architecture** | Distributed transactional nodes.21 | Serverless execution engine (Dremel).8 |
| **Scalability Focus** | Scalable transaction throughput (QPS).21 | Petabyte-scale scans and parallel processing.15 |

### **Zero-ETL Virtual Graphs and Reverse ETL Loops**

This integrated architecture is powered by federated queries and zero-ETL data sharing.8 By utilizing BigQuery's federated query engine and features like Spanner Data Boost, developers can query Spanner’s active operational tables directly from BigQuery without impacting transactional database performance.16  
This allows security teams to construct a **Virtual Graph** that merges real-time operational data with historical logs.16 For instance, a virtual graph can dynamically link live Account and Customer nodes residing in Spanner Graph with millions of historical LogIn and IP access edge tables stored in BigQuery.16 This enables immediate, multi-hop investigations that span both live data and historical logs without any data movement.16  
Once BigQuery Graph identifies historical fraud rings, the system feeds these insights back to the operational layer.16 Using a reverse ETL loop, newly discovered fraud markers (such as resolved device fingerprints or address hashes associated with a synthetic identity ring) are exported from BigQuery directly into Spanner Graph.16 This updates the operational blocklist, allowing Spanner Graph to intercept and block future registration attempts from that ring in real-time.16

## **Unstructured Data Ingestion and Agentic Reasoning in Fraud Environments**

Modern fraud systems must process both structured database tables and unstructured registration documents, such as physical photo IDs, utility bills, business registration certificates, and user-generated text.18 Google Cloud addresses this by integrating Document AI, machine learning embeddings, and generative AI functions directly within BigQuery Graph.8

\[Unstructured Files in Colossus\]   
               |  
               v  
   
               |  
               v  
 \[Gemini / Gemma Embedding Generation\]  \<-- (Autonomous Vector Sync)  
               |  
               v

               |  
               v  
   
               |  
               v  
 \[Looker / Conversational Agents\]

### **Unstructured Data Ingestion Pipeline**

To process unstructured files, the system implements an automated pipeline that extracts information and converts it into graph nodes 18:

* **Document Parsing**: Unstructured physical registration documents stored in Colossus are processed using the SQL function AI.PARSE\_DOCUMENT.15 This function automates Optical Character Recognition (OCR), layout parsing, and document chunking, converting unstructured PDF files or images into clean, structured text.22  
* **Entity Extraction**: Relational data, extracted textual entities, and their associations are processed using generative AI models (such as Gemini with Context Caching) to extract nodes and edge relations directly from the parsed document text.18  
* **Embedding Generation**: The system generates semantic vector representations of the extracted text using BigQuery-native Gemma embeddings.22 BigQuery's autonomous embedding generation fully manages this pipeline, automatically updating the vector index as new unstructured registration files are ingested.22

### **Hybrid Search and Vector-Graph Traversal**

By integrating semantic vector search with GQL traversals, platforms can identify highly sophisticated synthetic identities.5 BigQuery's hybrid search unifies semantic and full-text search into a single function, allowing the system to execute combined vector-graph queries.8  
This allows security systems to find "fraudster-like" accounts based on the semantic meaning of their registration documents, and then traverse their transactional paths within 1 to 6 hops to expose coordinated networks 8:

SQL  
GRAPH fraud\_demo.IdentityGraph  
MATCH (f:Customer)\--\>{1, 6}(target:Customer)  
WHERE VECTOR\_SEARCH(  
  (SELECT embedding FROM fraud\_demo.customer\_embeddings WHERE customer\_id \= f.customer\_id),  
  (SELECT embedding FROM fraud\_demo.blocked\_profiles)  
)  
RETURN target.customer\_id, target.creation\_time;

This hybrid approach allows platforms to identify synthetic profiles that share behavioral or semantic characteristics with known fraudulent networks, even if they have altered their physical identifiers to evade traditional database-level checks.3

### **Agentic Reasoning on the Business Map**

To make these complex networks accessible to business users, the architecture supports BigQuery Graph's native "measures".22 Measures allow developers to define and unify key analytical metrics and business rules directly within the logical graph structure.22  
This transforms raw graph data into a deterministic "business map".22 AI-driven Conversational Analytics Agents navigate this business map to trace the cascading impact of fraud patterns across the platform.22  
Because the business map is deterministic, conversational agents avoid the hallucinations common when querying raw database tables, delivering high accuracy and direct mathematical explainability.22  
Conversational agents perform dual reasoning: they can instantly calculate precise KPIs using defined measures while simultaneously traversing complex relationship paths to explain the "why" behind anomalous registration patterns.22  
Furthermore, this architecture integrates with Looker, allowing developers to define BigQuery Graphs directly within Looker with built-in validation and Git-based source control.22 This exposes the logical graphs as native Looker views, allowing analysts and investigators across the enterprise to query, reuse, and visualize the fraud network using standard business intelligence dashboards.22

## **Distributed Execution Mechanics and Query Plan Optimization**

BigQuery Graph's ability to execute massive, multi-hop traversals over billions of nodes is powered by BigQuery’s serverless, distributed analytics engine.8

### **Distributed Query Execution**

When a user executes a GQL query, the Dremel engine compiles the statement into a highly parallel physical execution graph.15

                 \+-----------------------------------------+  
                 |            Dremel Root Node             |  
                 |  (Receives GQL, resolves schemas, and   |  
                 |   compiles the query execution graph)   |  
                 \+--------------------+--------------------+  
                                      |  
                     \+----------------+----------------+  
                     |                                 |  
                     v                                 v  
          \+----------------------+          \+----------------------+  
          |     Dremel Mixer     |          |     Dremel Mixer     |  
          |  (Optimizes tasks &  |          |  (Optimizes tasks &  |  
          |   prunes partitions) |          |   prunes partitions) |  
          \+----------+-----------+          \+----------+-----------+  
                     |                                 |  
           \+---------+---------+             \+---------+---------+  
           v                   v             v                   v  
     \+-----------+       \+-----------+ \#\#\#\#\#\#\#\#\#\#\#\#\#       \+-----------+  
     | Leaf Slot |       | Leaf Slot | \# Shuffle   \#       | Leaf Slot |  
     | (Parallel |       | (Parallel | \# Network   \#       | (Parallel |  
     | Colossus  |       | Colossus  | \# (Jupiter) \#       | Colossus  |  
     | Scans)    |       | Scans)    | \#\#\#\#\#\#\#\#\#\#\#\#\#       | Scans)    |  
     \+-----------+       \+-----------+                     \+-----------+

During query compilation, the Dremel root node parses the GQL syntax, resolves the graph's physical metadata, and splits the execution plan into parallel sub-tasks.15 These tasks are distributed to intermediate nodes called *mixers*, which optimize the plan (for instance, pruning partitions to avoid scanning irrelevant historical dates) and distribute the operations to thousands of active leaf nodes, or *slots*.15  
Each leaf slot acts as an independent compute unit, reading its assigned slice of the node and edge tables from Colossus in parallel.15 The high-speed **Jupiter network** coordinates rapid data shuffling between slots during parallel joins and aggregations.15

### **Diagnostic Execution Steps**

Within the physical execution graph, the query plan is divided into distinct operational stages, each consisting of elementary execution steps 23:

* **READ**: Leaf slots read specific column shards from the node and edge tables stored in Colossus.15  
* **COMPUTE**: The slots evaluate the filters, inline expressions, and GQL property conditions defined in the query.23  
* **JOIN**: The engine correlates the node and edge tables to resolve the matched relationships.23  
* **AGGR**: The slots perform aggregations (such as counting the degree of a node) over the matched subgraphs.23  
* **REPARTITION / COALESCE**: Optimization steps applied directly to shuffled data to rebalance the workload evenly across worker nodes.23 This prevents single workers from becoming bottlenecks during computationally intensive graph joins.23  
* **WRITE**: The final results are compiled and written back to the user's workspace or returned to the client application.23

### **Execution Plan Optimization**

To prevent latency bottlenecks when querying massive graphs, developers can utilize BigQuery's built-in query plan diagnostics and performance insights 23:

* **Detecting Slot Contention**: High-depth graph queries (such as 5+ hops) require significant memory and compute resources, which can lead to slot contention in shared environments.8 To resolve this, teams can optimize their queries to use fewer resources, allocate dedicated slot reservations, or schedule resource-intensive community detection tasks during off-peak hours.25  
* **Data Input Scale Changes**: This diagnostic warning triggers when a query reads at least 50% more data than in previous runs.25 This is typically caused by rapid growth in the underlying tables.25 Developers should implement table partitioning (such as partitioning by registration timestamp) and clustering (such as clustering by device\_id or zip\_code) to ensure slots scan only the relevant data segments during the READ step.23  
* **Mitigating Data Skew**: Joining non-unique keys on both sides can cause data skew, where the output table is vastly larger than the input tables.25 Developers should carefully define their join and MATCH conditions, use inline filters to prune irrelevant nodes as early as possible, and avoid unconstrained, variable-length traversals that can cause combinatorial path explosions.18

## **Empirical Performance Outcomes, Case Studies, and Visual Forensics**

Deploying native graph architectures instead of legacy relational setups yields significant, measurable improvements in fraud prevention and operational efficiency.3

### **Enterprise Deployment Metrics**

Case studies across digital payment providers, telecommunications networks, and card-issuing fintechs show clear, quantitative improvements in detection rates and operational savings 3:

* **Virgin Media O2**: Replaced their legacy relational setup with BigQuery Graph, allowing them to run complex 4-hop queries that trace hidden connections across customer accounts, device fingerprints, and active networks.8 This setup acts as an early warning system, identifying and blocking registration attempts linked to known fraud networks before they can commit promotional abuse or cause financial damage.8  
* **Curve**: Saved approximately **£9.1M** by implementing BigQuery Graph for advanced, large-scale network analysis.8 The graph engine successfully maps and detects sophisticated, multi-layered fraud rings that were previously invisible to traditional SQL-based rules.8  
* **iuvity**: Transitioned to a native transaction graph architecture, scaling to over **250 million nodes** and **2.2 billion relationships**.17 The system processes approximately **500 transactions per second** with sub-millisecond query latencies.17 This graph approach **doubled the overall fraud detection rate** while maintaining a low false-positive rate, helping prevent over **$40 million** in annual fraud losses.17  
* **ArcadeDB Engine Benchmarks**: Achieved a **68% reduction in false-positive alerts** alongside a **35% increase in fraud detection accuracy**.3 The system resolved complex, 5-hop graph traversals with sub-50ms query response times, saving **$2.4 million annually** in fraud losses.3

### **Visual Diagnostics and Forensics**

A major challenge with automated machine learning and rule-based systems is the "explainability gap"—risk models output abstract probability scores, but fail to explain *why* an account was flagged as fraudulent.7 This gap makes manual audits, compliance reviews, and customer disputes difficult.7  
Graph query results address this by providing intuitive, visual explanations.7 The matched subgraph itself serves as clear evidence of fraud.7 Using BigQuery Studio notebooks, analysts can run GQL queries and render interactive, visual subgraphs directly using the %%bigquery \--graph visualization tool 8:

Python  
%%bigquery \--graph display\_only  
GRAPH \`my\_dataset.fraud\_demo\_graph\`  
MATCH p \= (c1:Customer)--\>(phone:Phone)\<--(c2:Customer)  
RETURN TO\_JSON(p) AS path

This query generates an interactive network visualization in the notebook.8 Analysts can drag, zoom, and select nodes to trace how accounts are connected.18

\[Customer A\] \------(HAS\_PHONE)------\> \[Phone Node: 555-0192\] \<------(HAS\_PHONE)------  
     |                                                                                     |  
 (REGISTERED\_ON)                                                                     (REGISTERED\_ON)  
     |                                                                                     |  
     v                                                                                     v  
 \<-------------------------------------------------------

This visualization makes the underlying fraud ring obvious.14 An investigator can instantly see that Customer A and Customer B are not isolated users; they are part of a coordinated registration attempt, sharing both a phone number and a hardware device fingerprint.3 This clear context helps analysts make fast decisions, resolve disputes, and maintain comprehensive audit trails for regulatory reporting.2  
Furthermore, for specialized visualization requirements, BigQuery Graph integrates with industry-leading partner platforms including G.V(), Graphistry, Kineviz, and Linkurious, allowing teams to explore and present complex query results outside the Google Cloud Console.8

## **Synthesis and Strategic Recommendations**

Based on the architectural requirements and performance metrics analyzed, organizations deploying graph technology to mitigate registration fraud should implement the following strategic guidelines:

* **Deploy a Hybrid Operational-Analytical Architecture**: Use Spanner Graph at the transactional edge to evaluate registration requests in real-time (sub-10ms), running fast, low-hop patterns to intercept known fraudulent indicators.16 Concurrently, stream registration data into BigQuery Graph to run resource-intensive community detection (WCC, Louvain) and multi-hop traversals over historical archives.8 Feed the outputs of these analytical runs back into Spanner's active blocklist using zero-ETL integration.16  
* **Implement Entity Resolution and Normalization Pre-Ingestion**: Clean and normalize registration fields (such as physical addresses and phone numbers) prior to graph ingestion.3 Combine database-native full-text fuzzy matching with graph patterns to resolve data variations, preventing fraudsters from bypassing detection using slight typographical edits.3  
* **Use Deterministic Graph Algorithms for Blocklisting**: Deploy Weakly Connected Components (WCC) to partition the registration network into disjoint communities.7 If a single profile in a community is confirmed as fraudulent, programmatically apply the risk label to all other accounts sharing that unique WCC community ID (wccId), preventing coordinated networks from executing attacks.13  
* **Optimize BigQuery Graph Layouts**: Ensure all underlying tables used to build nodes and edges are partitioned by date and clustered by key identifiers.23 Apply targeted inline filtering within GQL MATCH clauses to reduce the data scanned during execution, avoiding expensive cross-joins and slot contention.18  
* **Bridge the Explainability Gap with Visual Diagnostics**: Integrate BigQuery Graph with visualization tools (such as BigQuery Studio notebooks or partner platforms like Linkurious) to expose matched subgraphs to compliance teams.8 This provides human analysts with clear, visual evidence of fraud during manual reviews and regulatory audits.3

#### **Works cited**

1. TigerGraph Fraud Solutions \- Graph Database for Fraud Detection & Prevention, accessed May 28, 2026, [https://www.tigergraph.com/solutions/fraud-detection/](https://www.tigergraph.com/solutions/fraud-detection/)  
2. Fraud Detection: Leveraging Graph Databases \- Didit.me, accessed May 28, 2026, [https://didit.me/blog/automated-fraud-detection-with-graph-databases/](https://didit.me/blog/automated-fraud-detection-with-graph-databases/)  
3. Graph Database for Fraud Detection \- ArcadeDB, accessed May 28, 2026, [https://arcadedb.com/fraud-detection.html](https://arcadedb.com/fraud-detection.html)  
4. Community Detection \- TigerGraph, accessed May 28, 2026, [https://www.tigergraph.com/glossary/community-detection/](https://www.tigergraph.com/glossary/community-detection/)  
5. Introduction to BigQuery Graph \- Google Cloud Documentation, accessed May 28, 2026, [https://docs.cloud.google.com/bigquery/docs/graph-overview](https://docs.cloud.google.com/bigquery/docs/graph-overview)  
6. Knowledge Graph For Fraud Detection \- Meegle, accessed May 28, 2026, [https://www.meegle.com/en\_us/topics/knowledge-graphs/knowledge-graph-for-fraud-detection](https://www.meegle.com/en_us/topics/knowledge-graphs/knowledge-graph-for-fraud-detection)  
7. Graph Database for Fraud Detection: Prevent Fraud \- PuppyGraph, accessed May 28, 2026, [https://www.puppygraph.com/blog/graph-database-for-fraud-detection](https://www.puppygraph.com/blog/graph-database-for-fraud-detection)  
8. Introducing BigQuery Graph | Google Cloud Blog, accessed May 28, 2026, [https://cloud.google.com/blog/products/data-analytics/introducing-bigquery-graph](https://cloud.google.com/blog/products/data-analytics/introducing-bigquery-graph)  
9. Detecting fraud rings: the social-graph problem in disguise : r/SQL \- Reddit, accessed May 28, 2026, [https://www.reddit.com/r/SQL/comments/1tp8g0j/detecting\_fraud\_rings\_the\_socialgraph\_problem\_in/](https://www.reddit.com/r/SQL/comments/1tp8g0j/detecting_fraud_rings_the_socialgraph_problem_in/)  
10. Fraud Detection with BigQuery Graph \- Google Codelabs, accessed May 28, 2026, [https://codelabs.developers.google.com/codelabs/fraud-bigquery-graph](https://codelabs.developers.google.com/codelabs/fraud-bigquery-graph)  
11. What Is a Graph Database? \- Oracle, accessed May 28, 2026, [https://www.oracle.com/autonomous-database/what-is-graph-database/](https://www.oracle.com/autonomous-database/what-is-graph-database/)  
12. Graph-Based Approaches for Detecting Fraud Rings in Digital Platforms \- Medium, accessed May 28, 2026, [https://medium.com/@amistapuramk/graph-based-approaches-for-detecting-fraud-rings-in-digital-platforms-c3031f83ef99](https://medium.com/@amistapuramk/graph-based-approaches-for-detecting-fraud-rings-in-digital-platforms-c3031f83ef99)  
13. Exploring Fraud Detection with Graph Data Science (Part 2\) \- Neo4j, accessed May 28, 2026, [https://neo4j.com/blog/developer/exploring-fraud-detection-neo4j-graph-data-science-part-2/](https://neo4j.com/blog/developer/exploring-fraud-detection-neo4j-graph-data-science-part-2/)  
14. Create and query a graph | BigQuery \- Google Cloud Documentation, accessed May 28, 2026, [https://docs.cloud.google.com/bigquery/docs/graph-create](https://docs.cloud.google.com/bigquery/docs/graph-create)  
15. How Google BigQuery Works: Architecture, Components, and Query Execution \- Medium, accessed May 28, 2026, [https://medium.com/@nikhilmogre1998/how-google-bigquery-works-architecture-components-and-query-execution-db8b24c23946](https://medium.com/@nikhilmogre1998/how-google-bigquery-works-architecture-components-and-query-execution-db8b24c23946)  
16. The unified graph solution with Spanner Graph and BigQuery Graph | Google Cloud Blog, accessed May 28, 2026, [https://cloud.google.com/blog/products/data-analytics/the-unified-graph-solution-with-spanner-graph-and-bigquery-graph](https://cloud.google.com/blog/products/data-analytics/the-unified-graph-solution-with-spanner-graph-and-bigquery-graph)  
17. Detect fraud faster with a Transaction Graph \- Neo4j, accessed May 28, 2026, [https://neo4j.com/blog/fraud-detection/accelerate-fraud-detection-graph-databases/](https://neo4j.com/blog/fraud-detection/accelerate-fraud-detection-graph-databases/)  
18. BigQuery Graph Series | Part 3: Query and Visualize your Graph | by Rachael Deacon-smith | Google Cloud \- Medium, accessed May 28, 2026, [https://medium.com/google-cloud/bigquery-graph-series-2e35bb203aac](https://medium.com/google-cloud/bigquery-graph-series-2e35bb203aac)  
19. Graph query overview | BigQuery \- Google Cloud Documentation, accessed May 28, 2026, [https://docs.cloud.google.com/bigquery/docs/graph-query-overview](https://docs.cloud.google.com/bigquery/docs/graph-query-overview)  
20. GQL overview | BigQuery \- Google Cloud Documentation, accessed May 28, 2026, [https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/graph-intro](https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/graph-intro)  
21. Use BigQuery Graph and Spanner Graph \- Google Cloud Documentation, accessed May 28, 2026, [https://docs.cloud.google.com/bigquery/docs/graph-compare](https://docs.cloud.google.com/bigquery/docs/graph-compare)  
22. Unveiling new BigQuery capabilities for the agentic era | Google Cloud Blog, accessed May 28, 2026, [https://cloud.google.com/blog/products/data-analytics/unveiling-new-bigquery-capabilities-for-the-agentic-era](https://cloud.google.com/blog/products/data-analytics/unveiling-new-bigquery-capabilities-for-the-agentic-era)  
23. Query plan and timeline | BigQuery \- Google Cloud Documentation, accessed May 28, 2026, [https://docs.cloud.google.com/bigquery/docs/query-plan-explanation](https://docs.cloud.google.com/bigquery/docs/query-plan-explanation)  
24. BigQuery under the hood: Google's serverless cloud data warehouse, accessed May 28, 2026, [https://cloud.google.com/blog/products/bigquery/bigquery-under-the-hood](https://cloud.google.com/blog/products/bigquery/bigquery-under-the-hood)  
25. Understanding the BigQuery query execution graph | Google Cloud Blog, accessed May 28, 2026, [https://cloud.google.com/blog/products/data-analytics/understanding-the-bigquery-query-execution-graph](https://cloud.google.com/blog/products/data-analytics/understanding-the-bigquery-query-execution-graph)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAAaCAYAAABYQRdDAAAA9ElEQVR4Xu2Svw7BUBjFP2Ex2SxYiNFiwhuIQcILWOyMXkRi9g4WiQeQWGyIWRCJRWLw79zcpvS02ktHfskvbc65/XpzW5E/PqRgBw5g9iWvvNwbM4R3uII1mId9uIFlq/sI9cANJrgAPdH9nAs/LhK8C9U3OHzHUfQDcS6IoJfaFEQvXnLhgfHQq+jFXuf4NWqg8Q5MCTNUfQsXUdEDt1x4wC+OwCZlNiY7LcEWh36sRQ9Vu/ZC5TvKRvAAY5Q7UEPVz8+Di3BPWRLm4AzWqXMxludRnKxr27HCSdCRfUwVLjgMyxlm4ISLMHThFKa5+PPLPAAP3DTMgVek/AAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAaCAYAAAB7GkaWAAAAZ0lEQVR4XmNgGHigAMT30QVh4C0Q/0cXpAx0AnECuiAI/IDSIPsckSVmAjETlA2SdEWSY6iF0v0MeFwKkriALggCIgwQSTF0CRA4z4AwshyIpZHkwBI7oOxXyBIg4MwAUfAHXWLEAwCaDRQuuqoUtAAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAABHCAYAAAC6YRv5AAADf0lEQVR4Xu3dT8hlcxgH8CM0CVlo/KnxZ6RGVshKqNkPalhgrVgxIsVCLBCSlYiiWVhQiJWFsmFloZSNlLEgoRQbC+H39J7T+8wz971zmTnn3s77+dS3eZ7n3Pu+7+yezjn33K4DAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4OT21AEAAJthf8sNff1Jyz/pGAAAGyAWtPNLDwDABokF7cXSH0k9AAAbpp5hi/7Ovv4iHwAAYHrHWh5JfV7ezmx5LPVnpBoAgAnc33Iw9ed1WwvbCy0PpvlgOOsGAMAEfm3Zl/qbW67uTrw8Ori+DgAAGM/H3dZiljPI9eUtT7Rc1Pd3pGMAAKzR4ZYDZfZD6QEANtZz3Ylnpn7fPjxb8X89VIcAAJsmlpYLFsy+KrM5+rM7/p43AICNs9PN+C+1XFiHAABM6++Wb+qw93gdAAAwvZ3OrlXLHiz73Umyd/ulAAD8V6subGM9WHb4gMNcAwBwylZZKjxYFgBgjWJhi+/VzKL/ua89WBYAYM2OtPxVZr+U/vnSb4qxLtP+X9fUAQDA6fRqy9t12IuzcGM/WPbbbrXLs4P62mMLZmOLL5TPrmq5tMwAACYx1YNlV1243qyD3qrvP1V/dFu/6916oJvubwAAWItVl51Fr7ukWzwfy0fd4oXt3Jan6hAAYA5eb3mtr29puTEdqz6rg27r4b939/U9Lc+kY2PYaWELUy6OAACTGZacH1vOSn31cMtdddhtvz6+rP7a1I/FwgYA7Cpxo34sOXGv3CLfp/poy/7UD+L9P6X+nFT/luosvic1zuTtlGViYXuvDnsWNgBgduJSaCw5keGyaPZ0qt9quTL1g+H9j9YDzUN10Lui5fYlWSYWtvfrsGdhAwBmJxacr1MdYjEL9YG997UcLrNX+oTh/Zf1/4716dZY2D6sw56FDQCYnVhwLk51eLbl5ZbbWr7sZ4NPS58XpKHOs/g5p1t88OHzOuxZ2ACA2an3i92a6kXLT53VDyEcSPU7qZ7CnpZ76xAAYM7e6I5f4MIHpV8mlrsn63BEdZkEAJi96+qgt+pidHYdjCgu6+6tQwCA3eyBOlizm+oAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHa9fwGBYci/NFKI2gAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAZCAYAAAA4/K6pAAAAu0lEQVR4XmNgGAWjADvwBuL1QGyPLkEI1ADxfyB2hPI7gHgiQho/gGlmQhITBuJjSHy8AKT5KpTNCsRpUDF0cBKI+dAFQX4GKV4LxG1AnMEAsR0baEUXAIEUBuy2oQM3dAEYADkJZIAUmjg/EH+Dsq9A6V9QGgPsBuLXDBBNKkC8BYgXIMlHAHEWEG9HEsMAbEAcDcQy6BJQAHIlB7ogsYCZARFOScgSxAJdIF4FxHPQJUgBSugCo4BKAADh2Ry9cdAWzwAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAaCAYAAABRqrc5AAAA2ElEQVR4XmNgGNaAEYhV0QVJAU+B+D8UUwSuMFDBEJAB19AFSQUgQyLQBUkBUQyYXmkCYn80MbzgJgPCEC4gvg/EfED8Da6CCAAy4DYQCwLxRqjYT6g40QCkeCcQz0SXQAMnGSAuxACgwAQZchVK70GVRgGt6AIwcJ0B1dkg9hQkPlEAPX2A+Cuh7I9QWh6I9wHxXCgfA4A0haHxsxkgeekYVAwU6EJA/BemCBmIMWDGgB9U7AOaOChQKUqMIIBuGcnAgQHiJV4gFkCVIg38AOLl6IKjYDADAHSeMSpAayCmAAAAAElFTkSuQmCC>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABoAAAAZCAYAAAAv3j5gAAAA/0lEQVR4Xu2UMWoCYRCFn0UuIIKp7TyGN7DxDiaiqNh5gRReQyzSBISU3kFQsBELRRQtbLUwzjCr7IzzaxbSBPaDx7Lf/5hZ2GWBlP9Ih/Jm5RNKlBnlh9I3Z4oB5QQpct718UPalHPsvgqZ8ZSki7hfdNyHcXckWVSG//RH+F6RZNEI/sAFfK9IsugAf+AUvldwoWZlAO56A8fwvYILdSsDrOEPnMD3Ci40rAwQekdz+F7BhaaVAbrwB/76q2tZGVGh5I3jftZxQ+MUOUipZw+IDPyXv4F8zldeIZ2XmLvxSdlRVpRldN1CfktxviD/Qsse0v+GLCno45SUlL/gAnq9R8qMXCmBAAAAAElFTkSuQmCC>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADwAAAAaCAYAAADrCT9ZAAAClElEQVR4Xu2XOWhVQRSGj/uGgk1sXBAUO7HTKERUUEG0EEsxWAg22ohLJdhJqhQBLS0sBEEEG9FGbRIQQURwAwsRSdzFfUE9f85M3rz/nbu8NSLvg5935z9nZm7mzhaRLv8851R/2JxMtrLRYlapviTlNclz0yxTnVUNqRZQzOOg6gSbLeaC6iR5TX/xQbFG9oXyUtVL1beJjFqWqF6wmfBarM0o5qlUxx8lsV2q26pPITYriYHlqnfklWKqWIM3ORD4pfrNZgD1ZrNJXFbdEcvtpRiYrnpA3k7V86TsDRb4LJZbF2gMI53FZrGcLeSvV30nzwN1Z4bfHxQDh1Q7yEMuZg9YIdmzbKVkD4YLRrGoQpwBF8n/KeXW7ofwi3y0gy+aMkplkL7TedWppMwgdy6bHhvFkm+QzywUy3tPPrw55DGrVUfC8waxOrcq4XG8AU+92A/2Ew/EB9j0iCNetAb3iuXdTbz5wSvikmpaUkYdrjdMZYCN86HqutgpgD0g68S4JuWWltu5x2OxPHQc2RS8IjjndPDiV0ebzZ7hOEK5nxp6pPwf7OXtdzyPuH5T0vbG0kCDHJcS74JphqSvHCD2iOXxkdUf/DxwGzrKpnJfrG7dO2wG6KNUO96XY7Jy1onvp1yR2h0ZzBOri6/PG1gjnJHidxkHHeYlPhOLz+CAVHbuPPLiuMggjr2gWa5K9T07F3R6j03lldgungfq4kLhcVgsPoUDge2SPyD1gHZwLS5NvO+OiK1pPK+tyvBBXtxtI5jCH1VvgzDy26oyKrxho0HwHphxbeeY2MV+MlksrZsppUBn3sbUKfDf0m422wnW4hM2O8Qi8e/hbQf32ANsdoCOTmWmn40208dGly5d/l/+AqTzqj8xuKoBAAAAAElFTkSuQmCC>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAaCAYAAABVX2cEAAAA3klEQVR4XmNgGAWUgnlA/BmI/0PxAhRZCPjLgJAHYWdUaUyArBgb2AfEKuiC2AAjEG8H4vUMEMOCUKXBAJclGCAfiE2gbFyu+4MugAu8RWJ/YIAYxockpgbEnUh8vADZJaBwAfFvIoktA2IeJD5OAAqvzWhi6F7F5m2sADm8kMVABnRD+b+Q5PCCd+gCUABznTYQt6DJ4QS4vLCbASJ3D4g50eSwAhYg3osuCAVMDJhhhxMwA/EbID6JLoEEvgHxd3RBdLAKiD8yQNIXKF2B8h42oA/E2eiCo2AUDGkAAMruNN36aWNMAAAAAElFTkSuQmCC>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAZCAYAAADnstS2AAAAqUlEQVR4XmNgGDJAAohl0AXRwUIg/g/FRWhyWIEmA0QxC7oENrCSAaKYKABS+BVdEBn0AHETlA1SXIMkBweVQPwLylZlQHiOHa4CClKhEhxIYpegYhgAJPgci9h3NDEGD6hEOpo4SKwBTYxhMwOmdSlIYpZAzAWTSEOSgAGQR2FiH5ElQOA3EBcyQEz5wwAJGZBiZSBehKQODtQYIO6HASEgdkTijwI6AQCURSXAcD7IXAAAAABJRU5ErkJggg==>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADQAAAAaCAYAAAD43n+tAAACaklEQVR4Xu2W3YtNURjGH98UShqhkORKbkgmNfkMf4DiQiZFuSfJn6BIY5q4NHfyJyhxI7kSF4QkSUI+SiGf73PWWuPdz+y19zkz+xw351dP+6znffdaZ+397ndvoE9PuGL6o+b/Yq8aU8RvaIVpphtPmTWmy6ZR02KJlXHCdEbNyDLTWjUzcJ6b4k3rjl1EmOBIHK82vTV9m8iYzCrTazWNU6ZfCPNdkJhng+kFwhxvTNsLUWCu6bd4tfC2cuHbGoj8RH5SnjdfzcgAQpzHMlYibDqRuxt3TOfVrIITPVfTsQshZ7f420zfxfOMIf8nCTdz2I1zubOQj03iFeqT0x28Lv4P5J8dwnOq5vaxY8hXCGHuHjUV1isTb4mvLEHI+yQ+vQXieRhnc0kMydhv6CNCBTxynuex6Z6aCq8wJ809AwmWBfPuO29R9HLw+WCcR8KS3mgaR+ieZJPpvemJaYvpqWlzjClnUb1ei7qSSHBB5rGtJnZGL4d/SbJTkh3ROxjHnXAI1eu13g/tbqgs72iJ50nnaMtfKON22Yrq9SY6x1cNCAcQ8vSBHY5+DsaexSPb/tJiuGNYklXrtSi78kouZxDlPlmOEGMVkFSy04FlWjvHZ1QnvUSIz9EA/nW+Mi6hGLvmxiwd3vVO4esht14BJj1Q03iH0AWr4Ln8NFHSJ0/iqht/cH4nPESxy1bC1skF7yI8U/zNK1kH806qieCzaXjSK2K9+O3Cc/ep2TSnTV/U7AIz0Ga5NQEXmq1mw9wwjajZLfYjvOG7Bb8j/Rd5TzhnOq5mQ/R8M4lhNRpgnWmemn369Gmevw1OnHYhhbViAAAAAElFTkSuQmCC>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAbCAYAAACqenW9AAAAo0lEQVR4XmNgGNrgKhD/AeL/QMyJJocV7GWAKCYKgBSSpHg6uiA2IMUAUSyBLoENzGJAdUIbED9FE4MDZPceAmI+IF6FJIYCQIJzgfgSELNCxeYB8T24CiiQZECYnIMmhwEiGCAKQREDovegSqOC6wyobgOxpyDxUQBI8hoafyWU/RFJHAxAkmFo/GwgZgTiY0jiDGJQSWTgBxX7gCY+CgYbAADqfCrdk3T3XwAAAABJRU5ErkJggg==>