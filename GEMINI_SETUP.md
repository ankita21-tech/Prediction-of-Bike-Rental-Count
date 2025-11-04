# 🤖 Google Gemini AI Setup Instructions

## Getting Your Gemini API Key

1. **Visit Google AI Studio**
   - Go to: https://makersuite.google.com/app/apikey
   - Or visit: https://aistudio.google.com/app/apikey

2. **Sign in with Google Account**
   - Use your Google account to sign in

3. **Create API Key**
   - Click "Create API Key" button
   - Select "Create API key in new project" or use existing project
   - Copy your API key

4. **Set the API Key in Your Application**

   **Option 1: Environment Variable (Recommended)**
   ```bash
   # Windows PowerShell
   $env:GEMINI_API_KEY="your-api-key-here"
   
   # Windows Command Prompt
   set GEMINI_API_KEY=your-api-key-here
   
   # Linux/Mac
   export GEMINI_API_KEY="your-api-key-here"
   ```

   **Option 2: Update app.py directly**
   - Open `app.py`
   - Find this line (around line 14):
   ```python
   GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyDlbs0UfFGfsjDlfjdslfjdslfjdslkfjds')
   ```
   - Replace the placeholder key with your actual API key:
   ```python
   GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'YOUR-ACTUAL-API-KEY-HERE')
   ```

5. **Restart Flask Application**
   ```bash
   python app.py
   ```

## Features Enabled with Gemini AI

✅ **Intelligent Chatbot Responses**
- Natural language understanding
- Context-aware answers about bike rental predictions
- Smart suggestions and recommendations

✅ **Fallback System**
- If Gemini is unavailable, the system automatically uses built-in responses
- No interruption to user experience

## Testing the Chatbot

1. Login to your application
2. Navigate to the Chatbot page
3. You'll see "Powered by Google Gemini AI" badge
4. Try asking:
   - "What are the most important features?"
   - "How does weather affect bike rentals?"
   - "Tell me about the prediction models"
   - "What's the best season for bike rentals?"

## Troubleshooting

**Q: Chatbot not responding intelligently?**
- Check if API key is set correctly
- Verify internet connection
- Check console for error messages

**Q: "Gemini AI not configured" message in console?**
- API key is missing or invalid
- Set the environment variable or update app.py

**Q: Getting fallback responses instead of Gemini?**
- API quota might be exceeded (Gemini has free tier limits)
- API key might be invalid
- Check error messages in Flask console

## Free Tier Limits

- Google Gemini API has a generous free tier
- 60 requests per minute
- Sufficient for development and small production apps

## Security Note

⚠️ **Never commit your API key to version control!**
- Use environment variables
- Add `.env` file to `.gitignore`
- Use secrets management in production

---

Enjoy your AI-powered chatbot! 🚴✨
