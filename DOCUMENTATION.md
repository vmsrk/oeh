# Project Documentation: Email Triage Environment

## Overview
This project is a FastAPI-based server for an "Email Triage Environment." It provides endpoints to interact with an environment that simulates email triage tasks. The main components include:

- **FastAPI Server**: Handles HTTP requests and routes them to the appropriate environment methods.
- **EmailTriageEnv**: Represents the environment where email triage tasks are performed.
- **Endpoints**:
  - `/reset`: Resets the environment to a specific task.
  - `/step`: Takes an action in the environment and returns the result.
  - `/state`: Retrieves the current state of the environment.
  - `/health`: Checks the health of the server.
  - `/`: Root endpoint with a welcome message.

## Class Diagram
Below is the class diagram representing the relationships between the main components:

```
+-------------------+       +-------------------+
|   EmailTriageEnv  |<------|       Action      |
+-------------------+       +-------------------+
| - reset(task_id)  |       | - value           |
| - step(action)    |       +-------------------+
| - state()         |
+-------------------+
        ^
        |
+-------------------+
| FastAPI Endpoints |
+-------------------+
| - /reset          |
| - /step           |
| - /state          |
| - /health         |
| - /               |
+-------------------+
```

## Sequence Diagrams

### `/reset` Endpoint
```
Client -> Server: POST /reset { task_id }
Server -> EmailTriageEnv: reset(task_id)
EmailTriageEnv -> Server: Observation
Server -> Client: { observation }
```

### `/step` Endpoint
```
Client -> Server: POST /step { action }
Server -> EmailTriageEnv: step(action)
EmailTriageEnv -> Server: (observation, reward, done, info)
Server -> Client: { observation, reward, done, info }
```

### `/state` Endpoint
```
Client -> Server: GET /state
Server -> EmailTriageEnv: state()
EmailTriageEnv -> Server: State
Server -> Client: { state }
```

## How to Start

1. **Install Dependencies**:
   - Ensure Python is installed on your system.
   - Run the following command to install dependencies:
     ```
     pip install -r requirements.txt
     ```

2. **Run the Server**:
   - Start the FastAPI server by running:
     ```
     python server/app.py
     ```

3. **Access the API**:
   - Open your browser or API client (e.g., Postman) and navigate to `http://127.0.0.1:7860`.
   - Use the `/docs` endpoint to explore the API documentation.

4. **Test Endpoints**:
   - Use the `/reset`, `/step`, and `/state` endpoints to interact with the environment.

## Notes
- The `EmailTriageEnv` and `Action` classes are imported from `email_triage_env`. Ensure this module is correctly installed or available in the project.
- The server listens on port `7860` by default. You can change this in the `main()` function of `server/app.py`.

---

This documentation provides a high-level overview of the project and its components. For more details, refer to the codebase or the FastAPI auto-generated documentation at `/docs`. Happy coding!
