/**
 * Utility functions for handling authentication
 */

/**
 * Retrieves the authentication token from localStorage
 * @returns Promise<string> - The authentication token
 */
export const getAuthToken = async (): Promise<string> => {
  const token = localStorage.getItem('token');
  if (!token) {
    throw new Error('No authentication token found');
  }
  return token;
}; 