# Run MLPerf tests

NOTE: currently only tried with mixtral;
and only tried with offline benchmark

# How to run Offline
### 1. Install 

```
./install.sh
```

### 2. Run benchmark mode

```
./mixtral_run.sh performance
```

### 3. Accuracy mode

```
./mixtral_run.sh accuracy
```




# How to run Server

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

