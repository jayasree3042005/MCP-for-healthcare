# MCP-Based AI-Driven Healthcare Data Integration Platform

A healthcare interoperability platform that integrates **HL7/FHIR clinical data** and **DICOM medical imaging** into a unified web application.

The system provides patient-centered access to healthcare records, medical images, FHIR resources, conversion history, and AI-assisted interaction.

## Key Features

* HL7 message parsing and processing
* HL7 to FHIR data conversion
* FHIR patient and observation resource management
* DICOM medical image handling
* Patient-centered dashboard
* FHIR resource viewer
* DICOM image gallery
* Conversion history tracking
* AI-assisted healthcare chat
* Docker-based healthcare services

## Tech Stack

| Component       | Technology             |
| --------------- | ---------------------- |
| Backend         | Python, Flask          |
| Frontend        | HTML, CSS, JavaScript  |
| Clinical Data   | HL7, FHIR              |
| Medical Imaging | DICOM                  |
| Infrastructure  | Docker, Docker Compose |

## Project Structure

```text id="kzq75v"
MCP-for-healthcare/
│
├── docker/
│   └── docker-compose.yml
├── hl7/
│   └── parse_hl7.py
├── static/
├── templates/
├── uploads/
├── app.py
├── clinical_decision_support.py
├── clinical_response_formatter.py
├── healthcare_utils.py
└── requirements.txt
```

## Getting Started

### 1. Clone the Repository

```bash id="rlsl37"
git clone https://github.com/jayasree3042005/MCP-for-healthcare.git
cd MCP-for-healthcare
```

### 2. Start Docker Services

```bash id="ob7z8d"
cd docker
docker compose up -d
```

Verify that the containers are running:

```bash id="gohpso"
docker ps
```

Return to the project directory:

```bash id="o6rzou"
cd ..
```

### 3. Set Up Python Environment

Create a virtual environment:

```bash id="xb8g9j"
python -m venv venv
```

Activate it on Windows:

```bash id="p9tr3w"
venv\Scripts\activate
```

Install the dependencies:

```bash id="m7urc5"
pip install -r requirements.txt
```

### 4. Run the Application

```bash id="41l75n"
python app.py
```

Open the local URL displayed in the terminal to access the application.

## Application Modules

The web application provides interfaces for:

* Patient Dashboard
* FHIR Resources
* DICOM Gallery
* Conversion History
* AI Chat

## Docker Commands

**Start services**

```bash id="vvkpca"
docker compose up -d
```

**Check running containers**

```bash id="d02cny"
docker ps
```

**Stop services**

```bash id="z4hdba"
docker compose down
```

> Run the Docker Compose commands from the `docker` directory.

## Purpose

This project demonstrates the integration of different healthcare data standards into a single platform, enabling **HL7/FHIR clinical information** and **DICOM medical imaging** to be accessed through a unified patient-centered application.
