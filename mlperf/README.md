# Run MLPerf tests

NOTE: currently only tried with mixtral;
and only tried with offline benchmark

# How to run


### 1. Install 

```
./install.sh
```

## Offline runs:

To run accuracy mode
``` 
./llama_run.sh accuracy
```

To run performance mode
``` 
./llama_run.sh performance
```

Same for mixtral:
To run accuracy mode
``` 
./mixtral.sh accuracy
```

To run performance mode
``` 
./mixtral.sh performance
```




## ONline runs

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

