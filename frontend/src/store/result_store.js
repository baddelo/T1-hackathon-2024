import { writable } from 'svelte/store';

const resultStore = writable([]);
const resultImage = writable(null);

const setStoreResult = (newData) => {
  resultStore.update(() => (newData));
};

const setStoreImage = (newData) => {
	resultImage.update(() => (newData));
};

const clearStoreData = () => {
	setStoreResult([]);
	setStoreImage(null);
};

export {
	resultStore,
	resultImage,
	setStoreImage,
	setStoreResult,
	clearStoreData
};
