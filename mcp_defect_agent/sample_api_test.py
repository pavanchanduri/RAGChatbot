import requests
import json

def run_tests():
    results = []
    # Simulate a passing test
    try:
        resp = requests.get('https://httpbin.org/status/200')
        assert resp.status_code == 200
        results.append({'test_name': 'test_status_200', 'status': 'passed'})
    except Exception as e:
        results.append({'test_name': 'test_status_200', 'status': 'failed', 'error': str(e)})

    # Simulate a failing test
    try:
        resp = requests.get('https://httpbin.org/status/500')
        assert resp.status_code == 200
        results.append({'test_name': 'test_status_500', 'status': 'passed'})
    except Exception as e:
        results.append({'test_name': 'test_status_500', 'status': 'failed', 'error': str(e), 'stack_trace': 'Traceback (most recent call last)...'})

    # Send failures to defect agent
    for result in results:
        if result['status'] == 'failed':
            payload = json.dumps(result)
            # Replace with your API Gateway endpoint
            url = 'https://vr6acrk273.execute-api.us-west-2.amazonaws.com/Test/log-defect'
            print(f"Would POST to {url}: {payload}")
            resp = requests.post(url, data=payload, headers={'Content-Type': 'application/json'})
            print(resp.status_code, resp.text)

if __name__ == '__main__':
    run_tests()
