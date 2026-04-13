# Interview Questions & Expected Answers

---

## Troubleshooting

**Scenario:** Inexperienced developers wrote a web application (a Golang binary), deployed it on a Linux server, and went on a hiking trip. The application stopped responding (returns a white screen with HTTP 200 but no body). No access to source code — only root access is available.

**Where do you start diagnosing and what do you look at?**

**Expected Answer:**

1. **Check if the process is running:** `ps aux | grep <binary_name>` — confirm the process is alive.
2. **Check listening ports:** `ss -tlnp` or `netstat -tlnp` — verify the app is bound to the expected port.
3. **Check system resources:** `top` or `htop` — look for CPU/memory exhaustion, high load average.
4. **Check disk space:** `df -h` — a full disk can cause empty responses if the app can't write anything.
5. **Check logs:** look in `/var/log/`, `journalctl -u <service>`, or any app-specific log path. Even if logs are empty, that's informative.
6. **Check open file descriptors:** `ls /proc/<pid>/fd | wc -l` — FD exhaustion causes silent failures.
7. **Check network connections:** `ss -s` — look for connection queue saturation (SYN flood, too many TIME_WAIT).
8. **Try curl with verbose output:** `curl -v http://localhost:<port>/` — confirm the 200 with empty body, check headers for clues.
9. **Check OOM Killer history:** `dmesg | grep -i oom` — the process might be crashing and restarting.
10. **Strace as last resort:** `strace -p <pid>` — observe what syscalls the process is stuck on (blocked read, write to full disk, deadlock on mutex, etc.).

**Key insight:** HTTP 200 with no body usually points to the app panicking silently after headers are sent, a nil/empty response body being written, disk full preventing response construction, or a deadlock in a goroutine that handles the body write.

---

## Linux

### What should you pay attention to in the `top` output?

**Expected Answer:**

- **Load Average** (top-right): system load over 1, 5, 15 minutes — compare against CPU core count.
- **Tasks row**: total, running, sleeping, stopped, zombie — zombie count indicates broken process cleanup.
- **%Cpu(s) row**: `us` (user), `sy` (kernel), `id` (idle), `wa` (I/O wait). High `wa` means disk/network bottleneck. High `sy` means kernel overhead.
- **Mem/Swap rows**: total, free, used, buff/cache. Watch for swap usage — it means RAM is exhausted.
- **Per-process columns**: `PID`, `%CPU`, `%MEM`, `VIRT`, `RES`, `S` (state), `COMMAND`.
- **Process state `S`**: R=running, S=sleeping, D=uninterruptible sleep (I/O wait), Z=zombie, T=stopped.

---

### What do the three Load Average numbers mean, how are they calculated, and do processes in all states count toward them?

**Expected Answer:**

Load Average shows the average number of processes in the **run queue** (running or waiting to run) over the last **1, 5, and 15 minutes**. It is calculated as an exponentially weighted moving average.

A value equal to the number of CPU cores means the system is fully utilized. A value higher means there's a queue.

**What counts toward Load Average:**
- Processes in **R** state (running or runnable) — yes.
- Processes in **D** state (uninterruptible sleep, usually waiting for I/O) — **yes**, they count too.
- Processes in **S** (interruptible sleep), **Z** (zombie), **T** (stopped) — **no**.

This means high Load Average doesn't always mean high CPU usage — it could be a disk I/O bottleneck.

---

### What is a zombie process, how does it appear, and is it reflected in Load Average?

**Expected Answer:**

A **zombie process** is a process that has finished execution but still has an entry in the process table because its parent hasn't called `wait()` to read its exit status. It holds a PID and a minimal kernel record but consumes no CPU or memory.

It appears when:
- A child process exits.
- The parent process neglects to call `wait()` or `waitpid()`.

Zombies are **not** counted in Load Average. They are harmless in small numbers but can exhaust the PID table if the parent keeps spawning children without reaping them.

You can identify them in `top` with state **Z** or via `ps aux | grep Z`.

---

### Can a new process requiring 500 MB start if `free` shows only 370 MB, but several gigabytes are in `buff/cache`? Will the OOM Killer be invoked?

**Expected Answer:**

**Yes, it can start.** Linux uses a unified memory model where `buff/cache` is used for filesystem caching but is immediately **reclaimable** when an application needs memory. The kernel will evict cache pages to satisfy the allocation.

The `free` column in `free -h` (without `-/+ buffers/cache`) can be misleading — the **available** column is the more accurate indicator of how much memory can realistically be given to a new process.

**OOM Killer** will only be invoked if:
- All physical RAM is exhausted (including reclaimable cache).
- Swap is full or disabled.
- The new allocation still cannot be satisfied.

In this scenario: plenty of cache available → cache is reclaimed → process starts fine → no OOM Killer.

---

### What is swap, how does it work, and why did Kubernetes (until recently) require it to be disabled?

**Expected Answer:**

**Swap** is disk space used as an overflow area when physical RAM is exhausted. The kernel moves least-recently-used memory pages to swap (page out) and loads them back when needed (page in). It is much slower than RAM (milliseconds vs nanoseconds).

**Why Kubernetes required swap to be disabled:**

1. **Predictability:** Kubernetes resource management (requests/limits) assumes memory is a hard, fast resource. With swap, a pod exceeding its memory limit doesn't get OOM-killed — it just slows down dramatically, making behavior unpredictable.
2. **Scheduler assumptions:** The scheduler places pods based on declared requests. With swap, a node can appear to have available memory while actually being heavily swapping, leading to poor scheduling decisions.
3. **Latency:** Swapping causes severe latency spikes, violating SLOs for latency-sensitive workloads.

Since Kubernetes 1.28, **swap support was introduced as beta** with proper accounting, allowing it under controlled conditions.

---

### What is the difference between virtual (VIRT) and resident (RES) memory, and which column shows real consumption?

**Expected Answer:**

- **VIRT (Virtual Memory):** The total virtual address space reserved by the process — includes code, data, heap, stack, mapped files, shared libraries, and memory that has been `mmap`'d but not yet touched. It can be much larger than physical RAM.
- **RES (Resident Set Size):** The actual physical RAM currently used by the process — pages that are currently loaded in memory. This is the real consumption figure.
- **SHR (Shared):** The portion of RES that is shared with other processes (e.g., shared libraries).

**Look at RES** to understand how much RAM the process is actually consuming. For actual exclusive usage: `RES - SHR`.

---

### How can you view open file descriptors for a process without `lsof`?

**Expected Answer:**

Use the `/proc` filesystem:

```bash
# List all open FDs for a given PID
ls -la /proc/<pid>/fd

# Count them
ls /proc/<pid>/fd | wc -l

# See FD limit
cat /proc/<pid>/limits | grep "open files"

# Check FD details (what each points to)
ls -la /proc/<pid>/fd/
```

Each entry in `/proc/<pid>/fd/` is a symlink pointing to the actual file, socket, pipe, or device the FD refers to. Sockets show as `socket:[inode]`, which can be cross-referenced with `/proc/net/tcp`.

---

### If a log file filled the disk and the process keeps writing to it, how do you free space without restarting? What happens if you `rm` the file?

**Expected Answer:**

**If you `rm` the file:**
- The filename is unlinked from the directory, but the **inode and data blocks are NOT freed** as long as the process still holds an open file descriptor to it.
- The process will **not crash** — it continues writing to the same FD, which still points to the now-unlinked inode.
- Disk space is **not freed** until the process closes the FD (i.e., until restart).

**How to free space without restarting:**

Truncate the file through the process's open FD:

```bash
# Find the PID and FD number
ls -la /proc/<pid>/fd | grep <logfile>

# Truncate via the /proc FD path (no restart needed)
> /proc/<pid>/fd/<fd_number>
# or
truncate -s 0 /proc/<pid>/fd/<fd_number>
```

This zeroes the file while keeping the inode and FD intact — the process continues writing from offset 0 and disk space is freed immediately.

---

### What is the difference between symlink and hardlink? What is an inode?

**Expected Answer:**

**Inode:** A data structure on the filesystem storing metadata about a file: permissions, owner, timestamps, size, and pointers to the actual data blocks on disk. The inode does **not** store the filename.

**Hard link:** A directory entry that maps a filename directly to an inode. Multiple hard links can point to the same inode — the inode has a **reference count**. The data is only deleted when the reference count drops to zero (all hard links removed). Hard links cannot span filesystems and cannot link to directories.

**Symbolic link (symlink):** A special file that stores a **path string** pointing to another file or directory. It has its own inode. If the target is deleted, the symlink becomes dangling. Symlinks can span filesystems and can point to directories.

| Property | Hard Link | Symlink |
|---|---|---|
| Own inode | No (shares target's) | Yes |
| Works across filesystems | No | Yes |
| Can link to directory | No | Yes |
| Survives target deletion | N/A (is the target) | No (dangling) |

---

## Containerization and Docker

### How does virtualization differ from containerization?

**Expected Answer:**

**Virtualization (VMs):** A hypervisor (VMware, KVM, VirtualBox) emulates hardware and runs a full guest OS kernel. Each VM is fully isolated at the hardware level. Heavy — each VM carries a complete OS, typically hundreds of MB to GBs.

**Containerization:** Containers share the host OS kernel. Isolation is achieved via Linux kernel namespaces (PID, network, mount, UTS, IPC, user) and resource limits via cgroups. Containers are lightweight — only the application and its dependencies, typically MBs.

| | VM | Container |
|---|---|---|
| Kernel | Own guest kernel | Shared host kernel |
| Isolation | Hardware-level | OS namespace-level |
| Startup time | Minutes | Milliseconds |
| Overhead | High | Low |
| Security boundary | Strong | Weaker (shared kernel) |

---

### What Linux kernel mechanisms are used for container isolation besides cgroups?

**Expected Answer:**

- **Namespaces** — the primary isolation mechanism:
  - `pid` — isolates process IDs (container sees its own PID 1).
  - `net` — isolates network stack (own interfaces, routing tables, iptables).
  - `mnt` — isolates filesystem mount points.
  - `uts` — isolates hostname and domain name.
  - `ipc` — isolates inter-process communication (semaphores, message queues).
  - `user` — isolates user and group IDs (UID mapping).
  - `cgroup` — isolates the cgroup hierarchy view.
- **seccomp** — filters which syscalls a container can make.
- **AppArmor / SELinux** — mandatory access control profiles restricting file and capability access.
- **Linux Capabilities** — instead of full root, containers get a reduced set of capabilities (e.g., `CAP_NET_BIND_SERVICE` without `CAP_SYS_ADMIN`).
- **Overlay filesystem** — union mount filesystem enabling layered images (not isolation per se, but fundamental to container operation).

---

### Why do we need Docker if the Linux kernel can already create containers?

**Expected Answer:**

The Linux kernel provides the primitives (namespaces, cgroups) but using them directly requires writing complex code with `clone()`, `unshare()`, `mount()` syscalls and managing cgroup hierarchies manually.

Docker adds:
- **Image format and registry:** standardized, layered, portable images with a distribution ecosystem (Docker Hub).
- **Dockerfile:** declarative build system.
- **CLI and API:** simple `docker run`, `docker build`, `docker push` UX.
- **Layer caching:** efficient incremental builds.
- **Networking abstractions:** bridge networks, DNS between containers.
- **Volume management.**
- **Cross-platform consistency** (with a Linux VM under the hood on Mac/Windows).

In essence: the kernel provides the engine, Docker provides the car around it.

---

### Alpine and layer cache management

**Expected Answer:**

**Alpine Linux** is a minimal Docker base image (~5 MB) based on musl libc and busybox, compared to ~70 MB for Debian slim or ~200 MB for Ubuntu. Benefits: smaller attack surface, faster pulls, smaller final image. Downside: musl libc can have subtle compatibility differences with glibc-based software.

**Layer cache:** Docker builds images layer by layer. Each `RUN`, `COPY`, `ADD` instruction creates a layer. Layers are cached by their instruction + content hash. If nothing changes in a layer or above, Docker reuses the cached layer.

**Best practices for cache efficiency:**
- Put rarely-changing instructions first (e.g., `COPY go.mod go.sum ./` + `RUN go mod download` before `COPY . .`).
- Group `apt-get update` and `apt-get install` in a single `RUN` to avoid stale cache issues.
- Use `.dockerignore` to avoid cache busting from irrelevant file changes.

---

### How do you kill a Docker container from inside?

**Expected Answer:**

The container's lifecycle is tied to its **PID 1** process. When PID 1 exits, the container stops.

From inside the container:
```bash
# Send SIGTERM to PID 1
kill 1

# Or force kill with SIGKILL
kill -9 1

# If the shell is PID 1:
exit
```

Note: if PID 1 is a shell script that doesn't propagate signals, `kill 1` may not work as expected. Proper PID 1 management (using `exec` in entrypoint scripts, or tools like `tini`) is important.

---

### Where is Docker layer cache stored and how to reuse it on another machine?

**Expected Answer:**

**Local storage:** `/var/lib/docker/overlay2/` (with the overlay2 storage driver). Each layer is stored as a directory with its diff content.

**Reusing cache on another machine:**

1. **Push/pull via a registry:** push the image to a registry (Docker Hub, ECR, GCR). On another machine, `docker pull` will reuse matching layers already present locally.
2. **`--cache-from` flag:** `docker build --cache-from my-registry/myimage:latest .` — Docker pulls the image and uses its layers as cache source even if not locally present.
3. **BuildKit inline cache:** `docker buildx build --cache-to type=registry --cache-from type=registry` — exports and imports granular cache metadata to/from a registry.
4. **Export/import:** `docker save` / `docker load` for air-gapped environments.

---

### Dockerfile practice: flaws in a basic Go build Dockerfile, multi-stage builds

**Expected Answer:**

**Common flaws in a naive Dockerfile:**

```dockerfile
# BAD example
FROM golang:1.21
WORKDIR /app
COPY . .
RUN go build -o server .
CMD ["./server"]
```

Problems:
- Final image includes the full Go toolchain (~800 MB) — unnecessary for runtime.
- All source files are copied, including test files, `.git`, etc.
- No `.dockerignore`.
- `go mod download` not separated, so every code change re-downloads all dependencies.

**Good multi-stage Dockerfile:**

```dockerfile
# Stage 1: build
FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download                  # cached unless go.mod changes
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o server .

# Stage 2: minimal runtime image
FROM alpine:3.19
RUN apk --no-cache add ca-certificates
WORKDIR /root/
COPY --from=builder /app/server .
CMD ["./server"]
```

Benefits: final image is ~10-15 MB instead of 800 MB, no build tools in production, better cache utilization.

---

## Kubernetes

### Which components live in Control Plane vs Worker nodes?

**Expected Answer:**

**Control Plane (Master):**
- `kube-apiserver` — the API gateway; all communication goes through it.
- `etcd` — distributed key-value store; source of truth for all cluster state.
- `kube-scheduler` — watches for unscheduled pods and assigns them to nodes based on resources, affinity, taints.
- `kube-controller-manager` — runs control loops: node controller, replication controller, endpoint controller, etc.
- `cloud-controller-manager` — integrates with cloud provider APIs (load balancers, volumes, nodes).

**Worker Nodes:**
- `kubelet` — agent that communicates with the API server; ensures containers in pods are running as specified.
- `kube-proxy` — manages network rules (iptables/ipvs) for Service routing.
- **Container runtime** — containerd, CRI-O, or Docker (via shim).

---

### What are requests and limits?

**Expected Answer:**

- **Requests:** the amount of CPU/memory the scheduler **guarantees** to the container. Used by the scheduler to decide which node has enough room for the pod. The node's "allocatable" capacity is calculated based on sum of requests.
- **Limits:** the maximum CPU/memory a container can use. Enforced by cgroups at runtime.

**CPU:** requests/limits are in millicores (e.g., `500m` = 0.5 core). CPU is **compressible** — exceeding the limit causes throttling, not killing.

**Memory:** requests/limits in bytes. Memory is **incompressible** — exceeding the memory limit causes the container to be **OOM-killed** by the kernel.

---

### If requests are available on a node but pods exceeded limits, will the scheduler place a new pod there?

**Expected Answer:**

**Yes, the scheduler will place the pod there.** The scheduler only looks at **requests** vs the node's allocatable capacity — it does not consider actual current usage or whether existing pods are over their limits.

Limits are enforced at runtime by cgroups on the node, not by the scheduler. This can lead to a situation where the node is overcommitted in practice even though scheduling decisions appeared valid.

---

### Who controls limit compliance and who invokes OOM Killer?

**Expected Answer:**

- **CPU limits** are enforced by the **kernel's CFS (Completely Fair Scheduler)** via cgroup `cpu.cfs_quota_us`. Excess CPU is throttled — no killing.
- **Memory limits** are enforced by the **kernel's cgroup memory subsystem** (`memory.limit_in_bytes`). When a container exceeds its memory limit, the kernel's **OOM Killer** is invoked — it kills the process inside the container. The `kubelet` detects this and marks the container as `OOMKilled`, then restarts it according to the pod's `restartPolicy`.

The `kubelet` itself monitors pod resource usage via cAdvisor metrics but relies on the kernel for actual enforcement.

---

### How do you run a legacy app hardcoded to connect to Redis at `127.0.0.1`?

**Expected Answer:**

Place the application container and a Redis container **in the same Pod**. Containers within a pod share the same network namespace, meaning they share `127.0.0.1` (localhost). Redis listens on `127.0.0.1:6379`, the app connects to `127.0.0.1:6379` — it just works.

```yaml
spec:
  containers:
  - name: app
    image: myapp
  - name: redis
    image: redis:7
```

This is called a **sidecar pattern**.

---

### How do you scale such an app with HPA while keeping Redis shared?

**Expected Answer:**

This is the catch: if Redis is a sidecar in the pod, **each replica gets its own Redis instance** — they are not shared.

To keep Redis shared across all replicas:
1. **Deploy Redis as a separate Deployment/StatefulSet** with its own Service (e.g., `redis-service`).
2. **Use a network proxy sidecar** (e.g., `socat` or `envoy`) in each app pod that forwards `127.0.0.1:6379` to the Redis Service's ClusterIP:

```yaml
- name: redis-proxy
  image: alpine/socat
  args: ["TCP-LISTEN:6379,fork", "TCP:redis-service:6379"]
```

Now the app connects to `127.0.0.1:6379` → the sidecar proxy forwards to the shared Redis Service → all replicas share the same Redis. HPA scales the app pods, Redis remains a single shared instance.

---

### What is a Pod at the Linux kernel level?

**Expected Answer:**

A Pod is a group of containers sharing a set of **Linux namespaces**:
- A shared **network namespace** (same IP, same `lo` interface, hence shared `127.0.0.1`).
- A shared **IPC namespace** (can communicate via shared memory and semaphores).
- Optionally a shared **PID namespace** (containers can see each other's processes).

Each container still has its own **mount namespace** (separate filesystem).

Kubernetes implements this by first starting a special **pause container** (also called the "infra container") which creates and holds the namespaces. All other containers in the pod then join those namespaces via `clone()` flags.

---

### Why do containers in the same pod share `127.0.0.1`?

**Expected Answer:**

Because they share the same **network namespace**. A network namespace includes its own network interfaces, routing table, and loopback (`lo`) interface. All containers in a pod are attached to the same network namespace (created by the pause container), so they all see the same `lo` interface and the same `eth0`. Any process binding to `127.0.0.1` in one container is reachable via `127.0.0.1` from any other container in the same pod.

---

### Can containers of the same pod run on different physical nodes?

**Expected Answer:**

**No.** A pod is the atomic scheduling unit in Kubernetes. All containers of a pod are always scheduled together on the **same node**. This is a fundamental design constraint — containers in a pod share namespaces, which requires them to be co-located on the same host (you can't share a network namespace across physical machines without special tunneling).

---

### Differences between ReplicaSet, Deployment, StatefulSet, and DaemonSet?

**Expected Answer:**

| | ReplicaSet | Deployment | StatefulSet | DaemonSet |
|---|---|---|---|---|
| Purpose | Maintain N pod replicas | Manage ReplicaSets with rolling updates | Ordered, stateful pods | One pod per node |
| Pod identity | Interchangeable | Interchangeable | Stable (ordered names, sticky storage) | Per-node |
| Update strategy | Manual | Rolling update, rollback | Ordered rolling update | Rolling update per node |
| Persistent storage | No native support | No native support | PVC per pod (stable) | Varies |
| Use case | Rarely used directly | Stateless web apps, APIs | Databases, Kafka, ZooKeeper | Log collectors, monitoring agents, CNI plugins |

---

### What does Deployment add on top of ReplicaSet?

**Expected Answer:**

A Deployment manages a **history of ReplicaSets**. It adds:
- **Rolling updates:** creates a new ReplicaSet with the new spec and gradually scales it up while scaling down the old one.
- **Rollback:** `kubectl rollout undo` switches back to a previous ReplicaSet revision.
- **Revision history:** maintains `revisionHistoryLimit` old ReplicaSets for rollback.
- **Pause/resume:** ability to pause a rollout mid-way to check health.
- **Update strategies:** `RollingUpdate` (default) and `Recreate` (kill all, then start new).

You should almost never create a ReplicaSet directly — use Deployments.

---

### How to guarantee pods run on dedicated nodes using Taints, Tolerations, and Node Affinity?

**Expected Answer:**

**Taints** (on nodes) + **Tolerations** (on pods) — repel pods from nodes:
```bash
kubectl taint nodes <node> team=backend:NoSchedule
```
Only pods with a matching toleration can be scheduled on this node. But pods *with* the toleration can still be scheduled on *other* nodes too.

**Node Affinity** (on pods) — attract pods to specific nodes:
```yaml
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: team
          operator: In
          values: ["backend"]
```

**Combined approach for strict isolation:**
1. Label dedicated nodes: `kubectl label node <node> team=backend`
2. Taint dedicated nodes: `kubectl taint node <node> team=backend:NoSchedule`
3. Add toleration + node affinity to the team's pods.

Result: team's pods are attracted to their nodes (affinity) AND only they can land there (taint+toleration). Other teams' pods cannot schedule there due to the taint.

---

### What are finalizers in Kubernetes and how are they used?

**Expected Answer:**

A **finalizer** is a string key in `metadata.finalizers` of a Kubernetes object. It acts as a **pre-deletion hook** that prevents the object from being deleted until the finalizer is removed.

Workflow:
1. User runs `kubectl delete <resource>`.
2. API server sets `metadata.deletionTimestamp` but does **not** delete the object.
3. The controller responsible for the finalizer sees the timestamp, performs its cleanup logic (e.g., deletes external cloud resources, finalizes a volume, etc.).
4. Controller removes the finalizer from `metadata.finalizers` via a PATCH.
5. Once all finalizers are removed, the API server garbage-collects the object.

Common use cases: ensuring PersistentVolumes are deleted before PVCs, cleaning up cloud load balancers before a Service is deleted, custom operator cleanup logic.

---

### How does Pod Disruption Budget (PDB) work during updates?

**Expected Answer:**

A **PodDisruptionBudget** defines the minimum availability (or maximum unavailability) of a pod group during **voluntary disruptions** (node drains, rolling updates, cluster upgrades).

```yaml
spec:
  minAvailable: 2       # at least 2 pods must be up at all times
  # OR
  maxUnavailable: 1     # at most 1 pod can be down at a time
  selector:
    matchLabels:
      app: myapp
```

During a `kubectl drain` or a Deployment rolling update, the eviction API checks the PDB before evicting/replacing a pod. If removing the pod would violate the budget, the eviction is **blocked** until the condition is satisfied.

PDBs protect against accidental full outages during maintenance but only affect **voluntary** disruptions — node failures (involuntary) are not constrained by PDBs.

---

## Coding

**Problem:** Given a set of commits forming a branch or dependency graph, and a test that indicates whether a version is working or broken — identify the commit where the bug first appeared (state changes from "good" to "bad"). Minimize the number of test runs.

**Expected Answer:**

This is a classic **binary search on a sorted sequence** problem. It is exactly what `git bisect` implements.

**Algorithm:**
1. Establish a known-good commit and a known-bad commit.
2. Pick the **midpoint** commit in the range.
3. Run the test on the midpoint.
4. If **good** → the bug is in the upper half → move the lower bound up.
5. If **bad** → the bug is in the lower half (or at this commit) → move the upper bound down.
6. Repeat until the range narrows to a single commit.

**Complexity:** O(log n) test runs for n commits. For 1000 commits: ~10 test runs. For 1,000,000: ~20 runs.

**`git bisect` usage:**
```bash
git bisect start
git bisect bad HEAD          # current commit is broken
git bisect good v1.0.0       # this tag was known good
# git checks out midpoint automatically
# run your test, then:
git bisect good   # or
git bisect bad
# repeat until git reports the first bad commit
git bisect reset  # return to HEAD
```

**Automation:**
```bash
git bisect run ./run_test.sh   # git bisect runs the script automatically
```

**For a DAG (non-linear history):** binary search still applies on the topological order. `git bisect` handles merge commits by choosing a midpoint in the ancestor graph. In the worst case (many branches), the number of steps grows slightly but remains O(log n) on the number of testable commits.