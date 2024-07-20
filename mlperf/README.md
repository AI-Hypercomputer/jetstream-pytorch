# Run MLPerf tests

NOTE: currently only tried with mixtral;
and only tried with offline benchmark

# How to run

### 1. Install 

```
./install.sh
```

### 2. Start server

```
./start_server.sh
```

### 3. Warm up the server

```
python warmup.py
```

### 4. Run the benchmark, now it runs offline mode

```
./benchmark_run.sh
```

