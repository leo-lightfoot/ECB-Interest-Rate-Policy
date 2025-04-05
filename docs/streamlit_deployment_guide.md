# Streamlit Deployment Guide

This document outlines the deployment process for the ECB Interest Rate Policy Predictor application to Streamlit Cloud, including common issues and their solutions.

## Prerequisites

- A GitHub repository with your application code
- A Streamlit Cloud account (https://streamlit.io/cloud)
- The application's requirements listed in `requirements.txt`
- Python 3.8 or higher

## Deployment Steps

1. Ensure your repository structure includes:
   - `app.py` as the main Streamlit application file
   - `requirements.txt` with all dependencies
   - Necessary data files in `Processed_Data/`
   - Trained models in `models/` directory

2. Connect to Streamlit Cloud:
   - Log in to Streamlit Cloud
   - Link your GitHub repository
   - Select the repository and branch to deploy
   - Set the main file path to `app.py`
   - Configure any advanced settings if needed

## Common Issues and Solutions

### 1. Missing Dependencies

**Issue**: Application fails due to missing packages.

**Solution**: 
- Ensure all required packages are in `requirements.txt` with compatible versions
- Check for hidden dependencies of libraries you're using

### 2. Resource Limitations

**Issue**: Streamlit Cloud has resource limitations (memory, CPU).

**Solution**:
- Optimize your model loading (consider smaller models)
- Use caching (`@st.cache_data` and `@st.cache_resource`) for expensive computations
- Reduce the size of data files where possible

### 3. File Path Issues

**Issue**: File paths that work locally may not work in the cloud.

**Solution**:
- Use relative paths from the repository root
- Use `os.path.join()` for platform-independent paths
- Verify all data and model files are properly included in the repository

### 4. Timeout During Deployment

**Issue**: Deployment times out during dependency installation or initial startup.

**Solution**:
- Reduce the size of large dependency packages
- Use lazy loading for heavy components
- Split computation across multiple pages if applicable

### 5. Authentication and Secrets

**Issue**: Secure credentials or API keys need to be accessed.

**Solution**:
- Use Streamlit's secrets management
- Add a `.streamlit/secrets.toml` file locally (excluded from Git)
- Configure secrets in Streamlit Cloud dashboard

### 6. Browser Compatibility

**Issue**: Some UI elements may not work in all browsers.

**Solution**:
- Test your application in multiple browsers
- Use Streamlit's built-in components when possible
- Provide browser recommendations to users

### 7. Memory Leaks

**Issue**: Application consumes increasing memory over time.

**Solution**:
- Properly scope cached functions
- Clear unused variables
- Periodically restart the application (automatic in cloud environment)

## Best Practices

1. **Test Locally First**: Use `streamlit run app.py` to test your application locally before deployment.

2. **Version Control**: Specify version numbers for all dependencies in `requirements.txt`.

3. **Error Handling**: Add robust error handling to prevent application crashes.

4. **Logging**: Implement logging to help diagnose issues in production.

5. **Performance Monitoring**: Use Streamlit's built-in metrics to monitor application performance.

## Additional Resources

- [Streamlit Deployment Documentation](https://docs.streamlit.io/streamlit-cloud)
- [Streamlit Caching](https://docs.streamlit.io/library/advanced-features/caching)
- [Streamlit Secrets Management](https://docs.streamlit.io/library/advanced-features/secrets-management) 