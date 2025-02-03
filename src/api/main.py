from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel
from typing import List
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import date

# Initialize FastAPI app
app = FastAPI(
    title="Client Engagement API",
    description="API to manage and retrieve client engagement records.",
    version="1.0.0",
)

# Database connection details
DATABASE_CONFIG = {
    "dbname": "client_engagement",
    "user": "postgres",  # Replace with your PostgreSQL username
    "password": "000",  # Replace with your PostgreSQL password
    "host": "localhost",
    "port": "5432"
}

# Pydantic model for Client Engagement
class ClientEngagement(BaseModel):
    client_id: int
    client_name: str
    contact_email: str
    contact_phone: str
    signup_date: date
    engagement_type: str
    engagement_status: str
    last_meeting_date: date
    feedback_rating: int
    notes: str

    class Config:
        schema_extra = {
            "example": {
                "client_id": 1,
                "client_name": "Client A",
                "contact_email": "client_a@example.com",
                "contact_phone": "123-456-7890",
                "signup_date": date(2023, 10, 10),
                "engagement_type": "Consultation",
                "engagement_status": "Active",
                "last_meeting_date": date(2023, 10, 1),
                "feedback_rating": 4,
                "notes": "Very satisfied with the service."
            }
        }

# Function to connect to the database
def get_db_connection():
    try:
        conn = psycopg2.connect(**DATABASE_CONFIG, cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")

# API endpoint to fetch all client engagement records
@app.get(
    "/client-engagements",
    response_model=List[ClientEngagement],
    summary="Get all client engagements",
    description="Retrieve a list of all client engagement records."
)
def get_client_engagements():
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM client_engagement;")
        records = cursor.fetchall()
        return records
    except Exception as e:
        print(f"Error fetching records: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch records")
    finally:
        cursor.close()
        conn.close()

# API endpoint to fetch a single client engagement record by ID
@app.get(
    "/client-engagements/{client_id}",
    response_model=ClientEngagement,
    summary="Get a client engagement by ID",
    description="Retrieve a single client engagement record by its ID."
)
def get_client_engagement(
    client_id: int = Path(..., description="The ID of the client engagement to retrieve")
):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM client_engagement WHERE client_id = %s;", (client_id,))
        record = cursor.fetchone()
        if record:
            return ClientEngagement.parse_obj(record)
        else:
            raise HTTPException(status_code=404, detail="Client not found")
    except Exception as e:
        print(f"Error fetching record: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch record")
    finally:
        cursor.close()
        conn.close()