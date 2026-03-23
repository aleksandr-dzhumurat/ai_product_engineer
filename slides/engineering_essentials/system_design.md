# System Design


## Fundamental concepts in System Design interview

**PACELC theorem**: An extension of the CAP theorem that describes trade-offs in distributed databases. In case of network Partition (P), a system must choose between Availability (A) and Consistency (C). Else (E), when the system is running normally, it must choose between Latency (L) and Consistency (C).

**Heartbeat**: A mechanism used to monitor the health and availability of system components or nodes. Nodes periodically send signals (heartbeats) to indicate they're functioning properly. If a node fails to send heartbeats, the system can take corrective action like failover or resource reallocation.

**AJAX polling/HTTP short-polling**: A technique where the client repeatedly requests updates from the server at fixed intervals. The server immediately responds whether it has new data or not. This creates a "real-time-like" experience but can be resource-intensive due to many empty responses.

**HTTP long-polling**: An improvement over short polling where the server holds the client's request until new data is available or a timeout occurs. Once the client receives a response, it immediately sends another request. This reduces unnecessary server load compared to short polling. This technique is sometimes called **hanging GET**

**WebSockets**: A protocol providing full-duplex communication channels over a single TCP connection. Unlike HTTP, WebSockets maintain a persistent connection between client and server. This enables real-time, bidirectional data transfer with lower overhead than polling.

**Server-sent events (SSEs)**: A technology enabling servers to push data to web clients over HTTP. It's unidirectional (server to client only) but maintains a persistent connection. SSEs are ideal for real-time updates like news feeds or stock tickers.

## Fundamentals of distributed system

**Durability**: The guarantee that once data is committed, it will not be lost even in case of system failures. This is typically achieved through persistent storage and replication. Data durability is crucial for maintaining system reliability and data integrity.

**Replication**: The process of storing multiple copies of data across different nodes or locations. It improves data availability, fault tolerance, and read performance. Replication requires careful consideration of consistency models and update propagation strategies.

**Partitioning**: The division of large datasets across multiple nodes to improve scalability and performance. Also known as sharding, it allows horizontal scaling but introduces complexity in data distribution and query routing. Careful partition key selection is crucial for balanced data distribution.

**Consensus**: Algorithms and protocols that enable distributed systems to agree on shared values or states. Examples include Paxos and Raft, which ensure all nodes in a distributed system maintain consistent data. Consensus is fundamental for maintaining system consistency and reliability. Once we’ve achieved consensus, now transactions from applications need to be committed across databases (so-called distributed transactions), with fault checks performed by each involved resource. Two-way and three-way communication to read, write, and commit are shared across participant nodes.

## The architecture of large-scale web applications

**HTTP & REST**: HTTP is the foundation protocol for web communication, while REST is an architectural style for designing networked applications. REST uses standard HTTP methods (GET, POST, PUT, DELETE) and emphasizes stateless operations. Together they provide a standardized way to build scalable web services.

**Caching**: A technique to store frequently accessed data in faster storage tiers. Caching can happen at multiple levels (browser, CDN, application server, database) and significantly improves response times. Proper cache invalidation strategies are crucial for maintaining data consistency.

**CDNs**: Content Delivery Networks are distributed networks of servers that deliver content to users based on their geographic location. They reduce latency by serving content from edge locations closer to users. CDNs are essential for delivering static content and improving global application performance.

**N-Tier applications**: An architectural pattern that organizes applications into distinct functional layers (presentation, business logic, data). Each tier can be scaled independently based on specific requirements. This separation of concerns improves maintainability, scalability, and security.

**Stream processing** applies uniform processes to the data stream. If an application has continuous, consistent data passing through it, then stream processing allows efficient use of local resources within the application.

## Web Servers

For scalability, web servers are decoupled from application servers. **Web servers** are the first point of contact after load balancers. Data centers have racks full of web servers that usually handle API calls from clients. Depending on the service offered, the memory and storage resources in web servers can be small to medium. However, such servers require good processing resources. For example, Facebook has used a web server with 32 GB of RAM and 500 GB of storage space in the past.

## Application Servers

**Application servers** run the core application software and business logic. The difference between web servers and application servers is somewhat fuzzy. Application servers primarily provide dynamic content, whereas web servers mostly serve static content to the client. They can require extensive computational and storage resources. Storage resources can be volatile and nonvolatile. Facebook has used application servers with a RAM of up to 256 GB and two types of storage — traditional rotating disks and flash — with a capacity of up to 6.5 TB.

## Storage Servers

With the explosive growth of Internet users, the amount of data stored by giant services has multiplied. Additionally, various types of data are now being stored in different storage units. For instance, YouTube uses the following data stores:

1. **Blob storage** — used for its encoded videos.
2. **Temporary processing queue storage** — can hold a few hundred hours of video content uploaded daily for processing.
3. **Bigtable** — specialized storage for a large number of video thumbnails.
4. **RDBMS** — for users' and videos' metadata (comments, likes, user channels, etc.).

Other data stores are still used for analytics, for example, Hadoop's HDFS. Storage servers mainly include structured (SQL) and nonstructured (NoSQL) data management systems.

Returning to the example of Facebook: they've used servers with a storage capacity of up to 120 TB, housing exabytes of storage. However, the RAM of these servers is only 32 GB.

> By convention, storage and network bandwidth are measured in base 10, not base 2. One exabyte = $10^{18}$ bytes.

---

## Important Latencies

| Component | Time (nanoseconds) |
|---|---|
| L1 cache reference | 0.9 |
| L2 cache reference | 2.8 |
| L3 cache reference | 12.9 |
| Main memory reference | 100 |
| Compress 1 KB with Snzip | 3,000 (3 µs) |
| Read 1 MB sequentially from memory | 9,000 (9 µs) |
| Read 1 MB sequentially from SSD | 200,000 (200 µs) |
| Round trip within same datacenter | 500,000 (500 µs) |
| Read 1 MB sequentially from SSD ~1 GB/s | 1,000,000 (1 ms) |
| Disk seek | 4,000,000 (4 ms) |
| Read 1 MB sequentially from disk | 2,000,000 (2 ms) |
| Send packet SF → NYC | 71,000,000 (71 ms) |

---

## Request Estimation in System Design

This section estimates the number of requests a typical server can handle per second.

### CPU Time Formula

$$CPU_{\text{time per program}} = Instruction_{\text{per program}} \times CPI \times CPU_{\text{time per clock cycle}}$$

Where:
- $Instruction_{\text{per program}}$ — count of instructions the request consists of (no unit)
- $CPI$ — clock cycles required to process one instruction (no unit)
- $CPU_{\text{time per clock cycle}}$ — time the CPU takes to complete one clock cycle, in seconds

### Assumptions

- Clock rate: $3.5\ \text{GHz} = 3.5 \times 10^9$ cycles/second
- CPI = 1
- Instructions per request: $3.5 \times 10^6$

### Calculation

$$CPU_{\text{time per clock cycle}} = \frac{1}{3.5 \times 10^9}$$

$$CPU_{\text{time per program}} = (3.5 \times 10^6) \times 1 \times \frac{1}{3.5 \times 10^9} = 0.001\ \text{second}$$

$$\text{Total requests a CPU executes in 1 second} = \frac{1}{10^{-3}} = 1000\ \text{requests}$$

$$\text{Total requests a 64-core server executes in 1 second} = 64{,}000\ \text{requests}$$

> Note: changing assumptions (e.g. number of instructions per request) will produce different results. These are back-of-the-envelope calculations (BOTECs) — intentionally simplified, ignoring CPU/memory/IO-bound complexity.

# Microservice patterns

1. API Gateway Pattern: Centralizes external access to your microservices, simplifying communication and providing a single entry point for client requests.

2. Service Discovery Pattern: Enables microservices to dynamically discover and communicate with each other, simplifying service orchestration and enhancing system scalability.

3. Circuit Breaker Pattern: Implements a fault-tolerant mechanism for microservices, preventing cascading failures by automatically detecting and isolating faulty services.

4. Load Balancing Pattern: Distributes incoming traffic among multiple instances of a microservice, improving system performance, resilience, and resource utilization.

5. Bulkhead Pattern: Isolates microservices into separate partitions, preventing failures in one partition from affecting the entire system and enhancing system resilience.

6. CQRS (Command Query Responsibility Segregation) Pattern: Separates the read and write operations in a microservice, improving performance, scalability, and maintainability.

7. Event-Driven Architecture Pattern: Leverages events to trigger actions in your services, promoting loose coupling between services and enabling real-time responsiveness.

8. Saga Pattern: Manages distributed transactions across multiple microservices, ensuring data consistency while maintaining the autonomy of your services.

9. Retry Pattern: Enhances microservices' resilience by automatically retrying failed operations, increasing the chances of successful execution and minimizing transient issues.

10. Backends for Frontends Pattern (BFF): Creates dedicated backend services for each frontend, optimizing performance and user experience tailored to each platform.

11. Sidecar Pattern: Attaches additional components to your microservices, providing modular functionality without altering the core service itself.

12. Strangler Pattern: Facilitates the gradual replacement of a monolithic system with microservices, ensuring a smooth and risk-free transition.

# Design Patterns Used Across Different Software Layers

---

## Client Layer

**Client-side Load Balancing**: Distributes requests from the client directly to the available service instances.

**State Pattern**: Allows an object to alter its behavior when its internal state changes.

**Composite UI Pattern**: Composes responses from multiple microservices to render the complete UI.

---

## Load Balancer Layer

**Geographical Distribution**: Routes traffic based on the geographical location of the client.

**Health Checks**: Periodically checks the health of the servers and routes traffic only to healthy ones.

**Affinity Based Routing**: Routes the user's request to the same server for maintaining session persistence.

**Least Connections**: Routes traffic to the server with the fewest active connections.

---

## API Gateway Layer

**Backend for Frontend (BFF)**: Tailors API responses to the needs of individual client types.

**Circuit Breaker**: Detects failures and prevents applications from trying to perform actions that are doomed to fail.

**Retry Pattern**: Helps to handle transient failures when it tries to connect to a remote service or network resource.

**Request Collapsing**: Collapses multiple requests for the same operation into a single request.

---

## Web Server Layer

**Page Cache Pattern**: Stores the output of expensive operations and reuses it to avoid duplicated work.

**Compression Pattern**: Reduces the size of the response to improve load times.

**Lazy Loading**: Defers initialization of an object until the point at which it is needed.

**Content Negotiation Pattern**: The server generates different versions of a resource and serves the one matching the client's criteria.

---

## Application Server Layer

**Saga Pattern**: Manages long-running transactions and deals with failures and compensating transactions.

**CQRS (Command Query Responsibility Segregation)**: Separates read and write operations to improve performance and scalability.

**Proxy Pattern**: Provides a surrogate or placeholder for another object to control access to it.

**Chain of Responsibility**: Passes the request along a chain of handlers.

---

## Caching Layer

**Sidecar Caching**: Deploy a dedicated cache alongside each microservice to provide isolated and scalable caching functionality.

**Cache Chaining**: Arrange multiple cache layers hierarchically to handle different granularity or lifetime, querying each layer sequentially on a cache miss.

**Time-to-Live (TTL) Caching**: Assigns a predefined lifespan to each cache entry, removing or refreshing the entry once its lifespan expires.

---

## CDN Layer

**Prefetching**: Anticipates user actions and loads resources ahead of time.

**Parallel Requesting**: Makes multiple requests in parallel to improve load times.

**Edge Computing**: Processes data closer to the location where it is needed.

**Domain Sharding**: Splits resources across multiple domains to increase parallel downloads.

**Adaptive Image Delivery**: Delivers images tailored to the device and user context.

---

## Database Layer

**Sharding Pattern**: Distributes data across multiple databases to improve scalability.

**Replication Pattern**: Keeps copies of data in multiple locations for availability and reliability.

**Read-Replica Pattern**: Uses read replicas to offload read operations from the primary database instance.

**Query Object Pattern**: An object that represents a database query.


For elaboration, we’ll use a *“Requirements”* section whenever we design a building block (and a design problem). The *“Requirements”* section will highlight the deliverables we expect from the developed design. *“Requirements”* will have two sub-categories:

1. **Functional requirements:** These represent the features a user of the designed system will be able to use. For example, the system will allow a user to search for content using the search bar.
2. **Non-functional requirements (NFRs):** The non-functional requirements are criteria based on which the user of a system will consider the system usable. NFR may include requirements like high availability, low latency, scalability, and so on.

Availability

| Availability Percentages versus Service Downtime |  |  |  |
| --- | --- | --- | --- |
| **Availability %** | **Downtime per Year** | **Downtime per Month** | **Downtime per Week** |
| 90% (1 nine) | 36.5 days | 72 hours | 16.8 hours |
| 99% (2 nines) | 3.65 days | 7.20 hours | 1.68 hours |
| 99.5% (2 nines) | 1.83 days | 3.60 hours | 50.4 minutes |
| 99.9% (3 nines) | 8.76 hours | 43.8 minutes | 10.1 minutes |
| 99.99% (4 nines) | 52.56 minutes | 4.32 minutes | 1.01 minutes |
| 99.999% (5 nines) | 5.26 minutes | 25.9 seconds | 6.05 seconds |
| 99.9999% (6 nines) | 31.5 seconds | 2.59 seconds | 0.605 seconds |
| 99.99999% (7 nines) | 3.15 seconds | 0.259 seconds |  |

Reliability

Reliability (R) is the probability that the service will perform its functions for a specified time. R measures how the service performs under varying operating conditions.

We use two main metrics to measure R:

MTBF (Mean Time Between Failures)

```
MTBF = (Total Elapsed Time - Sum of Downtime) / Total Number of Failures
```

MTTR (Mean Time to Repair)

```
MTTR = Total Maintenance Time / Total Number of Repairs
```

> 💡 Note: We strive for a higher MTBF value and a lower MTTR value.
> 

Reliability (`R`) and availability (`A`) are two distinct concepts, but they are related. Mathematically, `A` is a function of `R`. This means that the value of `R` can change independently, and the value of `A` depends on `R`. Therefore, it’s possible to have situations where we have:

- low `A`, low `R`
- low `A`, high `R`
- high `A`, low `R`
- high `A`, high `R` (desirable)

**Scalability** is the ability of a system to handle an increasing amount of workload without compromising performance. A search engine, for example, must accommodate increasing numbers of users, as well as the amount of data it indexes.

The workload can be of different types, including the following:

- **Request workload**: This is the number of requests served by the system.
- **Data/storage workload**: This is the amount of data stored by the system.

**Vertical scaling**, also known as “**scaling up**,” refers to scaling by providing additional capabilities (for example, additional CPUs or RAM) to an existing device. Vertical scaling allows us to expand our present hardware or software capacity, but we can only grow it to the limitations of our server. The dollar cost of vertical scaling is usually high because we might need exotic components to scale up.

**Horizontal scaling**, also known as “**scaling out**,” refers to increasing the number of machines in the network. We use commodity nodes for this purpose because of their attractive dollar-cost benefits. The catch here is that we need to build a system such that many nodes could collectively work as if we had a single, huge server.

**Maintainability**, `M`, is the probability that the service will restore its functions within a specified time of fault occurrence. `M` measures how conveniently and swiftly the service regains its normal operating conditions.

For example, suppose a component has a defined maintainability value of 95% for half an hour. In that case, the probability of restoring the component to its fully active form in half an hour is 0.95. We use (mean time to repair) MTTR as the metric to measure `M`.

```python
MTTR = Total Maintenance Time / Total Number of Repairs
```

**Fault tolerance** refers to a system’s ability to execute persistently even if one or more of its components fail. Here, components can be software or hardware. Conceiving a system that is hundred percent fault-tolerant is practically very difficult.

## **Fault tolerance techniques**

### Replication

We create multiple copies of our data in separate storage. All copies need to update regularly for consistency when any update occurs in the data. Updating data in replicas is a challenging job. When a system needs strong consistency, we can synchronously update data in replicas. However, this reduces the availability of the system. We can also asynchronously update data in replicas when we can tolerate eventual consistency, resulting in stale reads until all replicas converge. Thus, there is a trade-off between both consistency approaches. We compromise either on availability or on consistency under failures—a reality that is outlined in the [CAP theorem](https://www.educative.io/answers/what-is-the-cap-theorem).

### **Checkpointing**

- **Consistent state:** A state is consistent in which all the individual processes of a system have a consistent view of the shared state or sequence of events that have occurred in a system. Snapshots taken in consistent states have data in coherent states, representing a possible situation of the system. For a checkpoint to be consistent, typically, the following criteria are met:
    - All updates to data that were completed before the checkpoint are saved. Any updates to data that were in progress are rolled back as if they didn’t initiate.
    - Checkpoints include all the messages that have been sent or received up until the checkpoint. No messages are in transit (in-flight) to avoid cases of missing messages.
    - Relationships and dependencies between system components and their states match what would be expected during normal operation.
- **Inconsistent state:** This is a state where there are discrepancies in the saved state of different processes of a system. In other words, the checkpoints across different processes are not coherent and coordinated.


The key principle is that a consistent checkpoint state should represent a possible global state of the system, where no message is received before it's sent in the recorded state. This is crucial for maintaining system integrity during recovery or state rollback operations.

### Case 1

You are leading a team developing a cloud-based file-sharing platform for a multinational corporation. You’re tasked with ensuring that the platform has the following attributes:

- Depending on the time and need, the platform is able to handle an increasing number of requests arriving from different parts of the world.
- The users of the platform have worldwide 24/7 access to its files.

Keeping these requirements in view, state ****two of the most important ****and relevant nonfunctional requirements from the list provided below such that their inclusion in the design will enable us to achieve the scenario mentioned above.

- Maintainability
- Scalability
- Fault tolerance
- Reliability
- Availability

### Answer

The two most important and relevant nonfunctional requirements for the cloud-based file-sharing platform are:

1. **Scalability** – This ensures that the platform can handle an increasing number of requests from different parts of the world as demand grows. Scalability will allow the system to dynamically allocate resources to maintain performance under varying loads.
2. **Availability** – Since users require worldwide 24/7 access to files, high availability is critical. This ensures minimal downtime and continuous access to the platform, even in the event of failures or maintenance.

## Case 2

Imagine a banking application for financial transactions and buying online products. This platform allows users to obtain their account status, transfer money, pay utility bills, and generate bank statements.

List the following **non-functional requirements in the correct order,** starting from the most important non-functional requirement to the least important non-functional requirement:

- Low latency
- Consistency
- Security

**Answer**

Certainly, let's prioritize the non-functional requirements for a banking application considering the scenario you described:

**1. Security:**

- **Importance:** Highest priority.
- **Explanation:** Banking applications deal with sensitive financial data (account balances, transaction history, personal information). Security breaches can have severe consequences for both the bank and its customers, including financial loss, identity theft, and reputational damage.

**2. Consistency:**

- **Importance:** Second highest priority.
- **Explanation:**
    - Inconsistent data can lead to incorrect account balances, unauthorized transactions, and disputes.
    - For example, if a user's account balance is not consistent across different systems, it can lead to overdrafts or declined transactions.
    - Maintaining consistency is crucial for ensuring accurate financial records and preventing errors in transactions.

**3. Low Latency:**

- **Importance:** Lower priority compared to security and consistency.
- **Explanation:**
    - While low latency is desirable for a smooth user experience, it is less critical than security and consistency in a banking context.
    - Short delays in transaction processing are generally acceptable as long as the system remains secure and consistent.
    - Excessive latency can impact user satisfaction and operational efficiency, but it is not as critical as maintaining the integrity and security of the financial data.

**Rationale:**

- Security is paramount in a banking application to protect sensitive financial information and prevent fraud.
- Consistency is essential for accurate financial record-keeping and to ensure that all transactions are processed correctly.
- While low latency is desirable for a smooth user experience, it can be compromised to some extent to prioritize security and consistency.

**Note:** This prioritization might vary slightly depending on the specific features and target audience of the banking application. For instance, a real-time payment system would likely place a higher emphasis on low latency.

Let me know if you have any other questions or would like to explore this further!

## Case 3

A space agency relies on critical systems to operate spacecraft, conduct space missions, and gather valuable data for scientific research. Imagine a scenario where a spacecraft is on a mission to explore a distant planet, and it encounters a hardware malfunction or a communication disruption with the control center on Earth.

State **one of the most important** **and relevant non-functional requirements** from the list provided below such that its inclusion in the design would enable us to recover from the scenario mentioned above. Please also provide proper **reasoning** behind your decision:

- Reliability
- Maintainability
- Consistency
- Fault tolerance
- Availability
- Scalability

**Note:** You must provide reasoning along with the name of the non-functional requirement in your answer.

## Answer

**Most Important Non-Functional Requirement:** **Fault Tolerance**

**Reasoning:**

In the scenario described, where a spacecraft encounters a hardware malfunction or communication disruption, **fault tolerance** is absolutely critical.

- **Fault tolerance** refers to the ability of a system to continue operating correctly even when encountering errors, failures, or unexpected conditions.
- In the context of a space mission, where the spacecraft is operating in a remote and hostile environment, the likelihood of encountering unforeseen issues is high.
- Fault tolerance measures would involve designing systems with redundancy, such as:
    - **Backup systems:** Having duplicate components (e.g., backup processors, communication links) that can take over if the primary components fail.
    - **Error detection and recovery mechanisms:** Implementing mechanisms to detect errors, isolate faulty components, and automatically switch to backup systems.
    - **Self-healing capabilities:** Enabling the spacecraft to autonomously diagnose and correct minor issues without human intervention.

By incorporating strong fault tolerance measures, the space agency can significantly increase the chances of mission success, even in the face of unexpected challenges.

**Why other options are less critical in this specific scenario:**

- **Reliability:** While important, reliability focuses on the overall dependability of the system over time. Fault tolerance is a more immediate and critical aspect when dealing with unexpected malfunctions or disruptions.
- **Maintainability:** Maintainability is crucial for long-term operations and upgrades, but it is less critical when dealing with an immediate emergency situation.
- **Consistency:** Consistency is important for data integrity, but it might be less critical in a life-or-death situation where immediate recovery and continued operation are paramount.
- **Availability:** Availability is related to fault tolerance, but fault tolerance specifically addresses the ability to continue operating correctly in the face of failures, whereas availability focuses on the overall uptime of the system.
- **Scalability:** Scalability is important for handling increased data volumes or user demands, which is not the primary concern in this specific scenario.

Therefore, fault tolerance is the most critical non-functional requirement to ensure the success of a space mission and the ability to recover from unexpected events.

## Case 4

Consider a scenario for Twitter’s recently introduced feature allowing users to stream live video directly within their tweets. As a result, the number of users streaming live video has surged exponentially. Users are consequently experiencing delays, and the system struggles to handle the increased load.

State **one of the most important** **and relevant non-functional requirements** from the list provided below such that its inclusion in the design would enable us to handle the scenario mentioned above.

- Reliability
- Maintainability
- Consistency
- Fault tolerance
- Availability
- Scalability

### Answer

**Most Important Non-Functional Requirement: Scalability**

**Reasoning:**

In the scenario described, where Twitter's live video feature experiences a surge in usage leading to delays and system overload, **scalability** is the most critical non-functional requirement.

- **Scalability** refers to the ability of a system to handle a growing amount of work or data without significant performance degradation.
- In this case, the system needs to be able to handle a significant increase in the number of users streaming live video simultaneously. This requires the system to:
    - **Increase its capacity:** Add more servers, storage, and network resources to accommodate the increased load.
    - **Adapt dynamically:** Adjust its configuration and resource allocation in real-time to handle fluctuations in traffic.
    - **Optimize resource utilization:** Efficiently utilize available resources to maximize throughput and minimize latency.

By incorporating strong scalability measures, Twitter can ensure that the live video feature can handle the growing number of users and provide a smooth, uninterrupted experience for all.

**Why other options are less critical in this specific scenario:**

- **Reliability, Maintainability, Fault Tolerance, Consistency:** While important for any system, these requirements are less critical in addressing the immediate issue of system overload due to increased usage.
- **Availability:** Availability is important to ensure the system is always accessible to users, but in this case, the primary issue is the system's ability to handle the increased load, not its overall availability.

Therefore, scalability is the most critical non-functional requirement to address the surge in live video usage and ensure a smooth user experience for Twitter's users.