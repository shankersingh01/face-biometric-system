/**
 * Image utility functions
 */

const LAST_REGISTERED_IMAGE_KEY = 'lastRegisteredImage';

export const getLastRegisteredImage = async (): Promise<string | null> => {
    try {
        // Get from localStorage
        const storedImage = localStorage.getItem(LAST_REGISTERED_IMAGE_KEY);
        if (storedImage) {
            console.log('Found last registered image in localStorage');
            return storedImage;
        }
        console.log('No last registered image found in localStorage');
        return null;
    } catch (error) {
        console.error('Error getting last registered image:', error);
        return null;
    }
};

export const setLastRegisteredImage = (imageUrl: string) => {
    try {
        // Store in localStorage
        localStorage.setItem(LAST_REGISTERED_IMAGE_KEY, imageUrl);
        console.log('Stored last registered image in localStorage');
    } catch (error) {
        console.error('Error storing last registered image:', error);
    }
};

export const clearLastRegisteredImage = () => {
    try {
        // Remove from localStorage
        localStorage.removeItem(LAST_REGISTERED_IMAGE_KEY);
        console.log('Cleared last registered image from localStorage');
    } catch (error) {
        console.error('Error clearing last registered image:', error);
    }
}; 