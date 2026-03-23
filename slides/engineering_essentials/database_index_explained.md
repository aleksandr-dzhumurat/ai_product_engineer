# DataBase index explained  

## B-tree vs LSTM tree

[B-tree vs LSTM tree](https://www.linkedin.com/posts/milanmilanovic_softwareengineering-programming-techworldwithmilan-activity-7348599306167189505-cmOD)

## Types of indexes

- B-Tree Index: Balanced tree structure for quick lookups, inserts, and deletes.Hash Index: Uses a hash table for fast exact-match queries.
- Bitmap Index: Efficient for columns with a limited number of distinct values, often used in data warehousing.
- Full-Text Index: Optimized for searching text within large documents.Clustered Index: Sorts and stores data rows in the table based on the index key.
- Non-Clustered Index: Separate structure from the data rows, includes a pointer to the actual data.
- Composite Index: Combines multiple columns into a single index to optimize queries involving those columns.
- Unique Index: Ensures all values in the indexed column(s) are unique.Covering Index: Includes all columns needed for a query, so the index alone can satisfy the query.
- Spatial Index: Optimized for querying spatial data, such as geographic coordinates.

## 𝗗𝗮𝘁𝗮𝗯𝗮𝘀𝗲 𝗜𝗻𝗱𝗲𝘅𝗶𝗻𝗴 𝗘𝘅𝗽𝗹𝗮𝗶𝗻𝗲𝗱

Most databases require some form of indexing to keep up with performance benchmarks. Searching through a database is much simpler when the data is correctly indexed, which improves the system's overall performance.

A database index is a lot like the index on the back of a book. It saves you time and energy by allowing you to easily find what you're looking for without having to flick through every page.

Database indexes work the same way. An index is a key-value pair where the key is used to search for data instead of the corresponding indexed column(s), and the value is a pointer to the relevant row(s) in the table.

To get the most out of your database, you should use the right index type for the job.

The 𝗕-𝘁𝗿𝗲𝗲 is one of the most commonly used indexing structures where keys are hierarchically sorted. When searching data, the tree is traversed down to the leaf node that contains the appropriate key and pointer to the relevant rows in the table. B-tree is most commonly used because of its efficiency in storing and searching through ordered data. Their balanced structure means that all keys can be accessed in the same number of steps, making performance consistent.

𝗛𝗮𝘀𝗵 𝗶𝗻𝗱𝗲𝘅𝗲𝘀 are best used when you are searching for an exact value match. The key component of a hash index is the hash function. When searching for a specific value, the search value is passed through a hash function which returns a hash value. That hash value tells the database where the key and pointers are located in the hash table.

𝗕𝗶𝘁𝗺𝗮𝗽 𝗶𝗻𝗱𝗲𝘅𝗶𝗻𝗴 is used for columns with few unique values. Each bitmap represents a unique value. A bitmap indicates the presence or absence of a value in a dataset, using 1’s & 0’s. For existing values, the position of the 1 in the bitmap shows the location of the row in the table. Bitmap indexes are very effective in handling complex queries where multiple columns are used.

When you are indexing a table, make sure to carefully select the columns to be indexed based on the most frequently used columns in WHERE clauses.

A 𝗰𝗼𝗺𝗽𝗼𝘀𝗶𝘁𝗲 𝗶𝗻𝗱𝗲𝘅 may be used when multiple columns are often used in a WHERE clause together. With a composite index, a combination of two or more columns are used to create a concatenated key. The keys are then stored based on the index strategy, such as the options mentioned above.

Indexing can be a double-edged sword. It significantly speeds up queries, but it also takes up storage space and adds overhead to operations. Balancing performance & optimal storage is crucial to get the most out of your database without introducing inefficiencies.

### Key Concepts in Data Modeling:

1. **Data Structures:**
    - **Tables (Relational):** Organizing data into rows and columns for relational databases.
    - **Documents (NoSQL):** Storing data in document-like structures (e.g., JSON, BSON) for flexible schemas.
    - **Graphs:** Representing relationships between entities using nodes and edges.
2. **Schema Design:**
    - **Conceptual Model:** A high-level representation of data entities and their relationships, often created during the initial project planning phase.
    - **Logical Model:** More detailed, specifying entities, attributes, and relationships without worrying about physical storage.
    - **Physical Model:** Defines how data is actually stored in a specific database (e.g., indexes, partitions).
3. **Normalization and Denormalization:**
    - **Normalization:** Organizing data to reduce redundancy and dependency by dividing data into multiple related tables.
    - **Denormalization:** Combining tables to improve query performance, commonly used in analytical systems.
4. **Data Types and Constraints:**
    - Specifying data types (e.g., integers, strings, dates) and constraints (e.g., primary keys, foreign keys, uniqueness) for each attribute to ensure data integrity.
5. **Dimensional Modeling:**
    - Used in data warehouses to design **star schemas** or **snowflake schemas** for efficient querying. It separates data into facts (measurable events) and dimensions (descriptive attributes).