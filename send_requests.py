def send_sample_request():
   import requests
   prompt = "How do I cook fried rice?"
   sample_input = {"prompt": prompt}
   output = requests.post("http://localhost:8000/default", json=sample_input)
   for line in output.iter_lines():
      print(line.decode("utf-8"))

send_sample_request()
