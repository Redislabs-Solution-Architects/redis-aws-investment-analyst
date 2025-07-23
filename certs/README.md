# TLS Certificates (Required)

Download your Redis Enterprise TLS certificates and rename to these exact file names:

- `redis_ca.pem` - CA certificate
- `client-cert.pem` - Client certificate  
- `client-key.pem` - Client private key

**Download from Redis Cloud console:**
1. Database → Security tab → Download certificates
2. Rename downloaded files to exact names above
3. Place in this `certs/` folder

**Note**: Required for Bedrock knowledge base integration. Files are git-ignored.