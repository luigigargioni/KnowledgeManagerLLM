# KnowledgeManagerLLM

## Creating a virtual environment 
It is higly suggested to create a virtual environment where all the dependencies are installed.
The command for the creation of a venv in python is:
```
python -m venv venv_name
```
where `venv_name` is the path in which the venv will be created.

Once it has been created the venv needs to be activated. Depending on the platform the command changes.

**Linux**
```
source venv/bin/activate
```

**Windows**
```
venv\Scripts\activate
```
Note that on windows the command could be rejected due to execution policies. To grant execution of scripts for one terminal session please run:
```
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
## Installing requirements
The project uses `pyproject.toml` file to indicate which libraries needs to be installed. 
To install all the required packages, do run, while having a venv activated and in the root folder of this project, the command:
```
pip install -e .
```

## Creating the .env file
This project uses .env files to store and retrieve environment variables. The file `.env.example` provides all the variables that need to be set for the program to work.
Before running the program do make a copy of the file and rename it `.env`.

## Running the application
Before running the application, two separate services are needed:
- **PostgreSQL database server**:  a `PostgreSQL` database should be already running and its credentials should be configured inside the `.env` file. Please refer to the [PostgreSQL documentation](https://www.postgresql.org/) to learn how to install it on your system.
- **Ollama backend**: to deploy local LLM the application uses the `Ollama` API. Make sure to install the backend and download the model you plan of using in the application. The model to use is specified in the `.env` file and must match the one installed in the system. Moreover, the model needs to support the endpoint `/api/chat`.

Note that the application may require a manual activation of the Ollama services using the command:
```
ollama serve
```

### Running through terminal
The `main.py` contains the code to run the system in the terminal. Hence it can be run with the command:
```
python ./main.py
```
while being in the folder `/src/`

### Running Streamlit web interface
The application also offers a web interface thanks to the `Streamlit` library.
To run the application with streamlit do use:
```
streamlit run ./chat_interface.py
```
while being in the folder `/src/`