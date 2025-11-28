from motor.motor_asyncio import AsyncIOMotorClient

# --- MongoDB Configurations ---
MONGODB_URL = "mongodb+srv://amrut989:amrut989@cluster0.akao778.mongodb.net/"
DATABASE_NAME = "AskMyPolicy"

class MongoDB:
    def __init__(self):
        self.client = None
        self.database = None

    async def connect(self):
        """Connect to MongoDB"""
        try:
            self.client = AsyncIOMotorClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
            self.database = self.client[DATABASE_NAME]
            await self.database.command("ping")
            print(f"✅ Connected to MongoDB database: {DATABASE_NAME}")
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            raise

    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            print("🔌 MongoDB connection closed")

# Global instance
mongodb = MongoDB()
