import os
import pytest
from app import app
import io

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_analyze_no_file(client):
    response = client.post('/api/analyze')
    assert response.status_code == 400
    assert b'No image provided' in response.data

def test_analyze_empty_file(client):
    data = {}
    data['image'] = (io.BytesIO(b''), '')
    response = client.post('/api/analyze', data=data)
    assert response.status_code == 400
    assert b'No selected file' in response.data

def test_analyze_invalid_file_type(client):
    data = {}
    data['image'] = (io.BytesIO(b'test'), 'test.txt')
    response = client.post('/api/analyze', data=data)
    assert response.status_code == 400
    assert b'Invalid file type' in response.data

def test_analyze_valid_image(client):
    # Create a small valid JPEG image
    data = {}
    data['image'] = (io.BytesIO(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00'), 'test.jpg')
    response = client.post('/api/analyze', data=data)
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'isReal' in json_data
    assert 'confidence' in json_data

def test_generate_no_file(client):
    response = client.post('/api/generate')
    assert response.status_code == 400
    assert b'No image provided' in response.data

def test_generate_empty_file(client):
    data = {}
    data['image'] = (io.BytesIO(b''), '')
    response = client.post('/api/generate', data=data)
    assert response.status_code == 400
    assert b'No selected file' in response.data

def test_generate_invalid_file_type(client):
    data = {}
    data['image'] = (io.BytesIO(b'test'), 'test.txt')
    response = client.post('/api/generate', data=data)
    assert response.status_code == 400
    assert b'Invalid file type' in response.data

def test_generate_valid_image(client):
    # Create a small valid JPEG image
    data = {}
    data['image'] = (io.BytesIO(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00'), 'test.jpg')
    response = client.post('/api/generate', data=data)
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'success' in json_data 