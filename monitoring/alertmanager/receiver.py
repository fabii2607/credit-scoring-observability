from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from datetime import datetime

class AlertHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        alerts = json.loads(post_data.decode('utf-8'))
        
        with open('/logs/alerts.jsonl', 'a') as f:
            for alert in alerts.get('alerts', []):
                alert['received_at'] = datetime.utcnow().isoformat()
                f.write(json.dumps(alert) + '\n')
                
        self.send_response(200)
        self.end_headers()

if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', 8000), AlertHandler)
    print("Receiver rodando na porta 8000...")
    server.serve_forever()