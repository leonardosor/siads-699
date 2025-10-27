# Use the official PostgreSQL image as the base
FROM postgres:latest

# Set environment variables for the PostgreSQL database
# These variables are used by the entrypoint script of the official image
ENV POSTGRES_DB=mydatabase
ENV POSTGRES_USER=myuser
ENV POSTGRES_PASSWORD=mypassword

# Expose the default PostgreSQL port (optional, but good for clarity)
EXPOSE 5432
