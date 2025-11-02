#!/bin/sh
# Start PostgreSQL
service postgresql start

# Wait a bit for PostgreSQL to come up
sleep 3

# Parse password and database name from DATABASE_URL
PW=$(echo "${DATABASE_URL}" | sed -E 's|.*://[^:]+:([^@]+)@.*|\1|')
DB=$(echo "${DATABASE_URL}" | sed -E 's|.*://[^@]+@[^:]+:[0-9]+/([^?]+).*|\1|')

# Alter user password
su - postgres -c "psql -c \"ALTER USER ${POSTGRES_USER} WITH PASSWORD '${PW}';\""

# Create database if it doesn't exist
if ! su - postgres -c "psql -tAc \"SELECT 1 FROM pg_database WHERE datname='${DB}'\""; then
  su - postgres -c "psql -c \"CREATE DATABASE \\\"${DB}\\\";\""
fi

# Enable pgvector extension
su - postgres -c "psql -d ${DB} -c \"CREATE EXTENSION IF NOT EXISTS vector;\""

# Now run the command passed to the container
exec "$@"
