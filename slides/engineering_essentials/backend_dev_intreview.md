# Interview Questions & Expected Answers

---

## Coding

### Implement a "Find Time" analog (like Outlook's scheduling assistant)

**Problem:**
- N people, each with a calendar (list of busy intervals)
- Meeting duration: M minutes
- Find the nearest time slot when all participants are free

**Expected Answer:**

**Approach:** merge all busy intervals across all participants, then scan gaps for a free slot ≥ M minutes.

```python
from typing import List, Tuple
import heapq

def find_meeting_slot(
    calendars: List[List[Tuple[int, int]]],  # list of busy intervals per person (in minutes from epoch)
    meeting_duration: int,
    search_start: int,
    search_end: int
) -> Tuple[int, int] | None:

    # 1. Collect and sort all busy intervals across all calendars
    all_busy = []
    for calendar in calendars:
        all_busy.extend(calendar)
    all_busy.sort(key=lambda x: x[0])

    # 2. Merge overlapping intervals
    merged = []
    for start, end in all_busy:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append([start, end])

    # 3. Scan gaps between merged intervals
    cursor = search_start
    for busy_start, busy_end in merged:
        if busy_start > cursor:
            gap = busy_start - cursor
            if gap >= meeting_duration:
                return (cursor, cursor + meeting_duration)
        cursor = max(cursor, busy_end)

    # 4. Check gap after last busy interval
    if search_end - cursor >= meeting_duration:
        return (cursor, cursor + meeting_duration)

    return None  # no slot found in search window
```

**Complexity:**
- Collecting intervals: O(N * K) where K = avg intervals per person
- Sorting: O(N·K · log(N·K))
- Merging + scanning: O(N·K)
- **Total: O(N·K · log(N·K))**

**Edge cases to mention:**
- Calendars with no busy intervals (empty list) — handled.
- Overlapping intervals within a single calendar — handled by merge.
- Meeting doesn't fit anywhere in the window — return None / signal no availability.
- Time zones — in a real system, normalize all times to UTC before processing.

**Follow-up: priority queue approach for streaming/large N:**

Use a min-heap of (next_busy_start, person_index, interval_index) to lazily merge calendars without loading all intervals into memory at once. Useful when calendars are fetched from an API page by page.

---

## Networking

### Why does TCP need timeouts? What is a half-open connection?

**Expected Answer:**

**TCP timeouts** exist because TCP is a stateful protocol — both sides maintain connection state. Without timeouts, a side that never hears back from the peer would hold resources (socket, buffers, PCB entry) forever.

Key TCP timers:
- **Retransmission timeout (RTO):** if an ACK is not received within RTO, the segment is resent. RTO is adaptive (based on RTT measurement).
- **Keep-alive timeout:** after prolonged silence, TCP sends probes to check if the peer is still alive.
- **TIME_WAIT (2·MSL):** ensures delayed packets from a closed connection don't corrupt a new connection on the same port.
- **FIN_WAIT_2 timeout:** prevents indefinite wait for a FIN from the peer.

**Half-open connection:**
A connection where one side believes the connection is established, but the other side has crashed or rebooted and lost state.

Example:
1. Client connects to server — 3-way handshake completes.
2. Server crashes and reboots (loses all TCP state).
3. Client is idle (no data to send).
4. Client still thinks the connection is open (ESTABLISHED state).
5. When the client eventually sends data, the server replies with RST (it has no record of this connection).
6. Until then, the client holds an open socket consuming resources.

**Fix:** TCP keep-alive probes (`SO_KEEPALIVE`) or application-level heartbeats detect this condition and close the stale socket.

---

### How does TLS work? What problems does it solve? What does a root certificate solve?

**Expected Answer:**

**TLS handshake (TLS 1.3 simplified):**
1. **ClientHello:** client sends supported cipher suites, TLS version, and a random nonce.
2. **ServerHello:** server picks cipher suite, sends its certificate (containing its public key) and its own nonce.
3. **Key exchange:** client and server run a key agreement protocol (ECDHE) — they each generate ephemeral key pairs, exchange public keys, and derive a shared secret without it ever being transmitted.
4. **Finished:** both sides derive symmetric session keys from the shared secret, send a Finished message encrypted with the new keys to verify everything.
5. **Application data:** encrypted with symmetric keys (AES-GCM, ChaCha20).

**Problems TLS solves:**
- **Confidentiality:** data is encrypted in transit — passive eavesdroppers see only ciphertext.
- **Integrity:** AEAD ciphers (e.g., AES-GCM) include a MAC — tampering is detected.
- **Authentication:** the server's certificate proves its identity — prevents impersonation.
- **Forward secrecy** (TLS 1.3, ECDHE): session keys are ephemeral — compromising the server's private key later doesn't decrypt past sessions.

**What root certificates solve:**
The chain of trust problem. How do you trust the server's certificate?
- Root CAs are self-signed certificates pre-installed in your OS/browser trust store.
- The server's certificate is signed by an Intermediate CA, which is signed by a Root CA.
- The client verifies the full chain up to a trusted root.
- This solves: "how to establish trust with a server you've never spoken to before, over an untrusted network."

Without root certificates, any attacker could present a fake certificate and intercept TLS connections (MITM).

---

### OSI model. Difference between TCP and UDP?

**Expected Answer:**

**OSI Layers:**

| Layer | Name | Example protocols |
|---|---|---|
| 7 | Application | HTTP, DNS, SMTP, gRPC |
| 6 | Presentation | TLS, encoding |
| 5 | Session | RPC sessions |
| 4 | Transport | TCP, UDP |
| 3 | Network | IP, ICMP, BGP |
| 2 | Data Link | Ethernet, ARP, MAC |
| 1 | Physical | Cables, fiber, WiFi signals |

**TCP vs UDP:**

| | TCP | UDP |
|---|---|---|
| Connection | Connection-oriented (3-way handshake) | Connectionless |
| Reliability | Guaranteed delivery, retransmission | No guarantee, no retransmission |
| Ordering | In-order delivery | No ordering guarantee |
| Error checking | Checksum + ACK | Checksum only |
| Flow/congestion control | Yes (sliding window, AIMD) | No |
| Overhead | Higher (headers, state) | Lower (8-byte header) |
| Latency | Higher | Lower |
| Use cases | HTTP, databases, file transfer | DNS, video streaming, gaming, VoIP |

**When UDP wins:** when occasional packet loss is acceptable and low latency matters more than reliability. Applications implement their own reliability if needed (e.g., QUIC is UDP + reliability layer).

---

### Difference between Unix socket and TCP/IP socket?

**Expected Answer:**

| | Unix Domain Socket | TCP/IP Socket |
|---|---|---|
| Transport | Kernel memory copy (no network stack) | Full TCP/IP network stack |
| Address | Filesystem path (e.g., `/var/run/app.sock`) | IP:port |
| Performance | ~2x faster, lower latency, less CPU | Higher overhead |
| Scope | Same machine only | Any machine (local or remote) |
| Authentication | Can use filesystem permissions + `SO_PEERCRED` (get peer PID/UID) | Relies on IP-level auth or application-level |
| Use cases | nginx ↔ PHP-FPM, PostgreSQL local connections, systemd socket activation | Any network communication, microservices |

**Key insight:** for same-host IPC, Unix sockets are strictly better than `127.0.0.1` TCP sockets — no TCP overhead, no loopback stack traversal. PostgreSQL uses Unix sockets by default for local connections.

---

### Difference between select/poll and epoll/kqueue?

**Expected Answer:**

All are I/O multiplexing mechanisms — they let a single thread monitor multiple file descriptors (sockets, files) for readiness.

**select/poll:**
- Pass the entire set of FDs to the kernel on every call.
- Kernel scans all FDs to find ready ones — O(N) per call.
- `select` has a hard FD limit (1024 by default). `poll` removes this.
- Stateless: the kernel has no memory of previous calls.
- Performance degrades linearly with the number of FDs.

**epoll (Linux) / kqueue (BSD/macOS):**
- Register interest in FDs once via `epoll_ctl` (EPOLL_CTL_ADD) — the kernel maintains the interest list.
- `epoll_wait` returns only the ready FDs — O(ready) instead of O(N).
- Scales to millions of concurrent connections.
- Supports edge-triggered (ET) and level-triggered (LT) modes.
  - LT (default): notifies as long as data is available (like select).
  - ET: notifies only when state changes (new data arrives) — requires draining the FD completely.

**Practical impact:** Node.js, Nginx, Go's netpoller, Redis — all use epoll/kqueue. This is what enables the C10K+ problem to be solved.

---

### What is a network interface? How does loopback work and what's its advantage?

**Expected Answer:**

**Network interface:** a software abstraction representing a network endpoint — either a physical NIC or a virtual/logical interface. Each has a name (`eth0`, `lo`, `docker0`), MAC address (except loopback), IP address, MTU, and TX/RX queues. Managed by the kernel's network stack.

**Loopback (`lo`):**
- A virtual interface with address `127.0.0.1` (IPv4) / `::1` (IPv6).
- Packets sent to loopback never leave the kernel — they are looped back entirely in kernel memory.
- No physical NIC, no driver interrupt, no network traversal.

**Advantages:**
- **Speed:** no serialization/deserialization, no wire — just a memory copy in the kernel.
- **Reliability:** no packet loss, no MTU issues, no physical layer failures.
- **Security:** not reachable from outside the host (unless explicitly routed).
- **Development/testing:** services can talk to each other without network configuration.

**Caveat:** still slower than Unix domain sockets, which bypass the TCP/IP stack entirely.

---

### When to use HTTP vs gRPC?

**Expected Answer:**

| | HTTP/REST (JSON) | gRPC |
|---|---|---|
| Protocol | HTTP/1.1 or HTTP/2 | HTTP/2 only |
| Payload | JSON (text) | Protocol Buffers (binary) |
| Schema | Optional (OpenAPI) | Mandatory (.proto) |
| Performance | Moderate | Higher (binary, multiplexing, compression) |
| Browser support | Native | Requires grpc-web proxy |
| Streaming | Limited (SSE, WebSocket workarounds) | Native (server/client/bidirectional) |
| Code generation | Optional | Built-in (strong types in all languages) |
| Debugging | Easy (human-readable JSON, curl) | Harder (binary, need grpcurl/Evans) |
| Ecosystem | Universal | Backend-to-backend |

**Use HTTP/REST when:**
- Public API consumed by browsers, mobile apps, or third parties.
- Simplicity and discoverability matter more than performance.
- Team is unfamiliar with Protobuf.

**Use gRPC when:**
- Internal microservice-to-microservice communication.
- Performance, low latency, or high throughput are critical.
- You need streaming (e.g., real-time data pipelines).
- Strong contract enforcement and code generation across multiple languages are valuable.

---

## Databases (PostgreSQL)

### What types of locks exist in PostgreSQL and at what levels?

**Expected Answer:**

**Lock levels:**
1. **Table-level locks** — acquired by DDL, bulk operations, explicit `LOCK TABLE`.
2. **Row-level locks** — acquired by DML (`SELECT FOR UPDATE`, `UPDATE`, `DELETE`).
3. **Page-level locks** — short-term, used internally during B-tree operations (rarely user-visible).
4. **Advisory locks** — application-level locks managed explicitly (`pg_advisory_lock`).

**Table-level lock modes (weakest → strongest):**

| Lock Mode | Acquired by | Conflicts with |
|---|---|---|
| ACCESS SHARE | `SELECT` | ACCESS EXCLUSIVE only |
| ROW SHARE | `SELECT FOR UPDATE/SHARE` | EXCLUSIVE, ACCESS EXCLUSIVE |
| ROW EXCLUSIVE | `INSERT`, `UPDATE`, `DELETE` | SHARE and stronger |
| SHARE UPDATE EXCLUSIVE | `VACUUM`, `CREATE INDEX CONCURRENTLY` | Itself and stronger |
| SHARE | `CREATE INDEX` (non-concurrent) | ROW EXCLUSIVE and stronger |
| SHARE ROW EXCLUSIVE | Rare | Most writes |
| EXCLUSIVE | Rare | All except ACCESS SHARE |
| ACCESS EXCLUSIVE | `DROP TABLE`, `TRUNCATE`, `ALTER TABLE`, `LOCK TABLE` | Everything |

**Row-level locks:** `FOR UPDATE`, `FOR NO KEY UPDATE`, `FOR SHARE`, `FOR KEY SHARE`. Used to prevent concurrent modification of selected rows.

---

### Why are long-running transactions bad? How do they affect vacuum, bloat, and performance?

**Expected Answer:**

**MVCC context:** PostgreSQL never overwrites rows in-place. An `UPDATE` inserts a new row version (tuple) and marks the old one as dead. `VACUUM` is responsible for removing dead tuples and reclaiming space.

**Problems with long-running transactions:**

1. **Vacuum is blocked:** `VACUUM` cannot remove dead tuples that are still visible to any active transaction (even read-only ones). A long `SELECT` started before an `UPDATE` prevents cleanup of all rows updated after it started.

2. **Table bloat:** dead tuples accumulate because vacuum can't clean them. The table file grows on disk, reads scan more pages, cache efficiency drops.

3. **Index bloat:** indexes also accumulate dead index entries pointing to dead heap tuples.

4. **XID wraparound risk:** PostgreSQL transaction IDs are 32-bit integers. Long-running transactions slow down XID advancement and can contribute to approaching the wraparound horizon (requiring emergency `VACUUM FREEZE`).

5. **Lock holding:** if the long transaction holds row locks, other transactions queue behind it, causing latency spikes.

6. **Replication lag:** on replicas, long-running queries on the standby can hold `hot_standby_feedback`, delaying vacuum on the primary.

**Mitigation:** set `statement_timeout`, `idle_in_transaction_session_timeout`, monitor `pg_stat_activity` for old transactions, use connection poolers (PgBouncer) to manage idle connections.

---

### How does MVCC work and why is it needed?

**Expected Answer:**

**MVCC (Multi-Version Concurrency Control)** allows readers and writers to operate concurrently without blocking each other.

**How it works in PostgreSQL:**
- Every row (tuple) has two hidden system columns: `xmin` (transaction ID that created it) and `xmax` (transaction ID that deleted/updated it, or 0 if current).
- When a transaction reads a row, it sees the version that was committed before its snapshot was taken — determined by its transaction ID and the visibility rules.
- `UPDATE` = insert new tuple (new `xmin`) + mark old tuple with `xmax` = current XID.
- `DELETE` = mark tuple with `xmax` = current XID. The data stays until vacuumed.

**Snapshot isolation:** each transaction gets a snapshot of committed data at its start (READ COMMITTED: per statement; REPEATABLE READ / SERIALIZABLE: per transaction). It only sees tuples where `xmin` is committed and ≤ snapshot XID, and `xmax` is either not committed or > snapshot XID.

**Why it's needed:**
- Without MVCC: reads would need to lock rows, blocking writes. High concurrency would cause severe contention.
- With MVCC: readers never block writers, writers never block readers. Long reads don't prevent inserts.

**Cost:** dead tuples accumulate and must be cleaned by VACUUM.

---

### Two parallel transactions inserting (A,B),(C,D) and (C,D),(A,B) — what problem can occur?

**Expected Answer:**

**Deadlock.**

Transaction 1: inserts (A,B) → acquires row lock on key A,B → then tries to insert (C,D) → waits for lock on C,D.
Transaction 2: inserts (C,D) → acquires row lock on key C,D → then tries to insert (A,B) → waits for lock on A,B.

Both transactions are waiting for each other → circular dependency → **deadlock**.

PostgreSQL detects deadlocks automatically via a deadlock detection algorithm. One of the transactions is chosen as the victim and rolled back with error: `ERROR: deadlock detected`.

**Prevention strategies:**
1. **Consistent ordering:** always acquire locks in the same order. If both transactions insert (A,B) first, then (C,D), there is no cycle.
2. **`SELECT FOR UPDATE` with ORDER BY** before batch operations.
3. **Retry logic** in the application for deadlock errors (SQLSTATE `40P01`).
4. **Reduce transaction scope** — shorter transactions reduce the window for deadlock.

---

### What indexes exist in PostgreSQL? When are they needed? Trade-offs?

**Expected Answer:**

**Index types:**

| Type | Best for | Notes |
|---|---|---|
| **B-tree** | Equality, range queries, ORDER BY, default | Works with `<`, `>`, `=`, `BETWEEN`, `LIKE 'prefix%'` |
| **Hash** | Equality only (`=`) | Faster than B-tree for pure equality; not replicated pre-PG10 |
| **GIN** | Full-text search, arrays, JSONB containment | Inverted index; slow to build, fast to query; good for `@>`, `@@` |
| **GiST** | Geometric data, full-text, range types, nearest-neighbor | Lossy (requires recheck); supports custom operators |
| **BRIN** | Very large tables with naturally ordered data (timestamps, sequential IDs) | Tiny index (stores min/max per block range); fast to build, imprecise |
| **SP-GiST** | Non-balanced structures (quadtrees, prefix trees) | Geometric, IP ranges |

**How to know an index is needed:**
- `EXPLAIN (ANALYZE, BUFFERS)` shows Seq Scan on a large table with high cost.
- Query filter column has high cardinality.
- Column appears in `WHERE`, `JOIN ON`, `ORDER BY`, `GROUP BY` frequently.
- Table is large (thousands+ rows); for tiny tables, seq scan is often faster.

**Trade-offs:**
- Indexes speed up reads but slow down writes (`INSERT`/`UPDATE`/`DELETE` must maintain all indexes).
- Indexes consume disk space and RAM (shared_buffers).
- Too many indexes → write amplification, bloat.
- Partial indexes (`WHERE active = true`) can be much smaller and faster for filtered queries.
- Covering indexes (`INCLUDE (col)`) avoid heap fetches for index-only scans.

---

### Index on (A, B) — is it useful for `WHERE B = 'val'`?

**Expected Answer:**

**Generally no** — not useful for this query.

A composite B-tree index on `(A, B)` stores entries sorted first by A, then by B within each A value. To find all rows where B = 'val', PostgreSQL would need to scan every A value's subtree — which degrades to a full index scan, often worse than a sequential table scan.

**The left-prefix rule:** a composite index `(A, B)` is efficiently used only when the query filters on A (the leading column), optionally also on B:
- `WHERE A = 'x'` ✅ — uses index.
- `WHERE A = 'x' AND B = 'y'` ✅ — uses index fully.
- `WHERE A > 'x'` ✅ — range scan on leading column.
- `WHERE B = 'y'` ❌ — cannot use the index efficiently (leading column not constrained).

**Solution:** create a separate index on `(B)` or `(B, A)` if queries on B alone are frequent.

**Exception:** if the query also selects A (index-only scan possible) and the planner estimates it's cheaper to scan the full index than the table, it might still use it — but this is rare and not reliable.

---

### How does PostgreSQL scale reads and writes? Limitations and typical approaches?

**Expected Answer:**

**Scaling reads:**
- **Streaming replication (physical):** create read replicas. Replicas are byte-for-byte copies of the primary, accepting read-only queries. Near-zero replication lag (async) or guaranteed consistency (synchronous, but adds write latency).
- **Connection pooling (PgBouncer):** PostgreSQL has expensive per-connection overhead (~5-10MB RAM). PgBouncer pools connections, allowing thousands of app connections over tens of DB connections.
- **Caching:** Redis/Memcached in front of the DB for hot read paths.
- **Partitioning:** horizontal table partitioning (range, list, hash) to reduce per-query scan size.

**Scaling writes — harder:**
- PostgreSQL has **no native multi-primary sharding**.
- **Vertical scaling:** bigger machine (more RAM for shared_buffers, faster NVMe).
- **Logical replication + sharding at application level:** route writes to different primaries by key range (Citus extension does this transparently).
- **Partitioning + tablespaces:** spread partitions across different disks/nodes.
- **Write batching and async commits:** `synchronous_commit = off` reduces fsync latency (risk: last few ms of committed transactions lost on crash).
- **Citus** (distributed PostgreSQL): transparent sharding for write scaling.

**Fundamental limits:**
- Single-node write throughput is bounded by WAL I/O.
- Vacuum must keep up with dead tuple accumulation.
- `max_connections` has hard limits (prefer pgBouncer in transaction mode).

---

### What problems must be considered with concurrent transactions?

**Expected Answer:**

**Standard concurrency anomalies (per SQL standard isolation levels):**

| Anomaly | Description | Prevented at |
|---|---|---|
| **Dirty read** | Reading uncommitted data from another transaction | READ COMMITTED+ |
| **Non-repeatable read** | Same row returns different values within a transaction | REPEATABLE READ+ |
| **Phantom read** | Same query returns different set of rows | SERIALIZABLE |
| **Lost update** | Two transactions read-modify-write the same row; one overwrites the other | REPEATABLE READ+ with locking |
| **Write skew** | Two transactions read overlapping data, each writes based on stale view | SERIALIZABLE only |

**PostgreSQL-specific issues:**
- **Deadlocks** — circular lock dependencies (see above).
- **Lock contention** — `FOR UPDATE` on hot rows causes queue buildup.
- **Serialization failures** (SQLSTATE `40001`) at SERIALIZABLE level — application must retry.
- **Sequence gaps** — `SERIAL`/`SEQUENCE` values are not rolled back on transaction abort.
- **Advisory lock misuse** — forgetting to release advisory locks causes starvation.

---

### Implement a correct money transfer between two accounts

**Expected Answer:**

Key requirements: atomicity, no lost updates, deadlock prevention.

```sql
-- Correct implementation with consistent lock ordering to prevent deadlocks
BEGIN;

-- Always lock accounts in consistent order (by ID) to prevent circular deadlocks
SELECT id, balance
FROM accounts
WHERE id IN (from_account_id, to_account_id)
ORDER BY id
FOR UPDATE;

-- Validate sufficient funds
-- (application checks result before proceeding)

UPDATE accounts
SET balance = balance - :amount
WHERE id = :from_account_id AND balance >= :amount;

-- Check that exactly one row was affected (insufficient funds → 0 rows)
-- If 0 rows affected: ROLLBACK

UPDATE accounts
SET balance = balance + :amount
WHERE id = :to_account_id;

-- Optional: insert audit record
INSERT INTO transfers (from_id, to_id, amount, created_at)
VALUES (:from_account_id, :to_account_id, :amount, NOW());

COMMIT;
```

**Key points:**
- **`FOR UPDATE` with `ORDER BY id`**: locks both rows before any modification, always in the same order — prevents deadlock regardless of which direction the transfer goes.
- **`balance >= :amount` in the UPDATE**: atomic check-and-deduct — avoids TOCTOU race between checking balance and deducting.
- **Single transaction**: both debits and credits are atomic — no partial transfers.
- **Isolation level**: READ COMMITTED is sufficient here due to explicit `FOR UPDATE` locking. REPEATABLE READ adds no benefit because we're locking anyway.
- **Audit log inside the transaction**: the transfer record is committed atomically with the balance changes.

---

## Kafka

### Topic vs Partition?

**Expected Answer:**

- **Topic:** a logical named stream of records — like a database table name. Producers publish to a topic, consumers subscribe to it.
- **Partition:** a topic is split into N partitions. Each partition is an **ordered, append-only log** stored on disk. Partitions are the unit of parallelism and distribution.

Key properties:
- Messages within a partition have a guaranteed order (by offset).
- Messages across partitions have NO order guarantee.
- Each partition is stored on one broker (with replicas on others).
- More partitions = more parallelism (more consumers can read in parallel) but more overhead (more files, more leader elections, more memory).
- **Partition key:** producers can specify a key — all messages with the same key go to the same partition (ordering guarantee per key).

---

### Cleanup policy: log compaction vs delete?

**Expected Answer:**

**`delete` (default):** segments are deleted based on `retention.ms` (time) or `retention.bytes` (size). Old messages are simply dropped. Use for event streams where historical data has a TTL.

**`log.compaction`:** Kafka retains only the **latest value per key** within a partition. Older records with the same key are removed during compaction (a background process). The latest record per key is always retained (even if very old). Records with a `null` value (tombstones) signal deletion of a key.

Use compaction for:
- **Changelog topics** (Kafka Streams state stores).
- **Event sourcing** snapshots.
- **CDC topics** — the topic acts as a compacted "current state" store.
- Consumer that joins late and needs the current state, not the full history.

`cleanup.policy=compact,delete` is also valid — combines both.

---

### Can you search for a message in Kafka?

**Expected Answer:**

**Not natively** — Kafka is not a database and has no query engine.

What you can do:
- **Seek by offset:** `consumer.seek(partition, offset)` — O(1) access to a specific offset if you know it.
- **Seek by timestamp:** `consumer.offsetsForTimes(timestamp)` — Kafka stores a timestamp index per segment, allowing binary search for the offset nearest to a given timestamp. O(log N) per partition.
- **Scan the partition:** consume from offset 0 and filter — O(N), only practical for small topics or debugging.

For actual message search: export to a search system (Elasticsearch, ClickHouse) or use **KSQL/Flink** to materialize a queryable state store.

---

### Consumer guarantees within a consumer group?

**Expected Answer:**

- **Each partition is assigned to exactly one consumer** in the group at any time — no two consumers in the same group process the same partition concurrently.
- **At-least-once delivery** (default): consumer commits offset after processing. If the consumer crashes before committing, messages are reprocessed after rebalance.
- **At-most-once:** commit offset before processing. If crash occurs, messages are skipped. Rarely desired.
- **Exactly-once:** achievable with Kafka Transactions (producer + consumer in the same transaction) or idempotent consumers.
- **Ordering:** guaranteed within a partition. No cross-partition ordering guarantee.
- **Total parallelism** in a group is bounded by the number of partitions — extra consumers beyond partition count sit idle.

---

### How to tune producer: `linger.ms` and `batch.size`?

**Expected Answer:**

Kafka producers batch messages before sending to brokers to improve throughput.

- **`batch.size`** (default: 16KB): maximum bytes per batch per partition. If the batch fills up before `linger.ms` expires, it is sent immediately. Increase for higher throughput at the cost of memory.
- **`linger.ms`** (default: 0ms): how long the producer waits to fill a batch before sending even if `batch.size` is not reached. `0` = send immediately (low latency, small batches). Higher values = larger batches, better compression ratio, higher throughput, more latency.

**Tuning for throughput:**
```
linger.ms=10-50
batch.size=65536 (64KB) or higher
compression.type=lz4 or snappy  # compress batches
acks=1 or all (depending on durability requirements)
```

**Tuning for low latency:**
```
linger.ms=0
batch.size=16384 (default)
acks=1
```

Also relevant: `buffer.memory` (total producer memory buffer), `max.in.flight.requests.per.connection` (pipelining), `retries` and `retry.backoff.ms`.

---

### What is a rebalance and how does it affect the group?

**Expected Answer:**

A **rebalance** is the process of reassigning partition ownership among consumers in a group. It is triggered by:
- A consumer joining the group.
- A consumer leaving or crashing (heartbeat timeout: `session.timeout.ms`, default 10s).
- A new partition being added to the topic.
- A group coordinator failure.

**Impact:**
- During a rebalance, **all consumers in the group stop processing** (stop-the-world) until the new assignment is complete. This is called the "rebalance storm" in large groups.
- Any uncommitted offsets at rebalance time may be reprocessed (at-least-once semantics).
- Long processing loops can trigger rebalance if `max.poll.interval.ms` is exceeded (consumer appears dead).

**Mitigation:**
- **Incremental Cooperative Rebalancing** (default since Kafka 2.4 with `CooperativeStickyAssignor`): only moves partitions that need to change owner — consumers continue processing their retained partitions during rebalance.
- Tune `session.timeout.ms` and `heartbeat.interval.ms` (heartbeat should be ~1/3 of session timeout).
- Increase `max.poll.interval.ms` if processing is slow.
- Use **static group membership** (`group.instance.id`): a consumer with a known ID has a grace period before its partitions are reassigned, surviving brief restarts without triggering rebalance.

---

### How to atomically write to both DB and Kafka? Outbox vs CDC

**Expected Answer:**

The **dual-write problem:** writing to DB and Kafka in two separate calls is not atomic — a crash between them leaves them out of sync.

**Pattern 1: Transactional Outbox**
1. In the same DB transaction as the business write, insert a record into an `outbox` table.
2. A separate **relay process** polls the outbox, publishes events to Kafka, marks records as sent.
3. Guarantees: the event is written if and only if the DB transaction commits. At-least-once delivery (relay may publish twice if it crashes after publishing but before marking sent) — deduplicate with idempotent consumers.
- Pros: simple, no external tooling.
- Cons: polling adds latency, relay is an extra process to maintain.

**Pattern 2: CDC (Change Data Capture)**
1. Leverage the DB's **replication log** (PostgreSQL WAL, MySQL binlog).
2. A CDC tool (Debezium) reads the log and publishes changes to Kafka.
3. No application code changes needed. DB is the source of truth.
- Pros: no outbox table, near-real-time, captures all changes including those not in application code.
- Cons: operational complexity (Debezium, Kafka Connect), schema change handling, initial snapshot complexity.

**Which to choose:**
- Outbox: simpler setup, good for greenfield.
- CDC: better for existing systems, high-volume, or when you can't modify the application.

---

### Delivery semantics: exactly-once, at-least-once. Why is exactly-once hard?

**Expected Answer:**

- **At-most-once:** produce/consume without retries. Messages may be lost on failure. Simple, lowest overhead. Use for metrics/logs where loss is acceptable.
- **At-least-once:** retry on failure + commit offsets after processing. Messages may be duplicated. Most common default. Requires idempotent consumers.
- **Exactly-once:** each message is processed exactly once, even in the presence of failures. Hardest to achieve.

**Why exactly-once is hard:**

In a distributed system, you cannot distinguish between "the message was processed but the ack was lost" and "the message was never processed." The network is unreliable.

Kafka's approach to exactly-once (EOS):
1. **Idempotent producer** (`enable.idempotence=true`): the broker deduplicates retried produce requests using a producer ID + sequence number. Prevents duplicate writes from producer retries.
2. **Transactional API** (`transactional.id`): producer wraps a batch of produce calls in a transaction. Either all writes (across multiple partitions/topics) commit or none do.
3. **Read-process-write with transactions**: consumer + producer in the same transaction (Kafka Streams does this). Offsets are committed atomically with output records.

**Limitations of EOS in practice:**
- Only within Kafka-to-Kafka pipelines (Kafka Streams). If you write to an external system (DB, API), you're back to at-least-once unless the external system supports idempotent writes.
- Higher latency and overhead.
- `isolation.level=read_committed` required on consumers to not read uncommitted transactional records.

---

### Kafka lag is growing — what do you do? How do you tune consumption?

**Expected Answer:**

**Step-by-step diagnosis:**

1. **Measure the lag:** `kafka-consumer-groups.sh --describe --group <group>` — see lag per partition, consumer host, last committed offset.
2. **Identify the bottleneck:**
   - Is lag uniform across all partitions? → Consumer throughput is the bottleneck.
   - Is lag only on some partitions? → Consumer imbalance, hot partitions, or a stuck consumer.
   - Is production rate spiking? → Temporary traffic burst, may self-resolve.
3. **Check consumer logs** for errors, slow processing, GC pauses, downstream system timeouts.
4. **Check consumer CPU/memory** — GC thrashing, thread contention.
5. **Check downstream system** (DB, API) — if processing involves DB writes, the DB may be the bottleneck.

**Tuning levers:**

- **Add consumers / scale out:** up to the number of partitions — beyond that, consumers sit idle. If you need more parallelism, increase partition count first.
- **`max.poll.records`** (default 500): reduce if processing each batch is slow (prevents `max.poll.interval.ms` timeout); increase if processing is fast (better throughput).
- **`fetch.min.bytes` / `fetch.max.wait.ms`**: increase to fetch larger batches per poll, reducing round trips.
- **`max.partition.fetch.bytes`** (default 1MB): max bytes fetched per partition per request — increase for higher throughput, watch memory.
- **`max.poll.interval.ms`**: increase if processing takes longer than the default 5 minutes (prevents spurious rebalances).
- **Async / parallel processing within consumer:** fetch synchronously, process records in a thread pool, commit offsets after all threads complete. Greatly increases throughput but complicates offset management.
- **Batching downstream writes:** instead of one DB write per message, accumulate a batch and insert in bulk.
- **Increase partition count** (requires topic recreation or partition reassignment) for more consumer parallelism.
- **Consumer group lag alerting:** set up alerts on consumer lag (via Burrow, Prometheus kafka_exporter) before it becomes critical.