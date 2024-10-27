<script>
	import { Button, Card } from 'flowbite-svelte';

	import { setStoreResult, setStoreImage } from '../../../store/result_store.js';

	import Uploader from '../Uploader/index.svelte';

	let value = [];
	let loading = false;

	// const fetchUrl = 'http://192.168.43.86:8000/detect';
	const fetchUrl = 'http://backend:8000/detect';

	const handlePostDetect = async () => {
		loading = true;
		
		const formData = new FormData();
		formData.append('image', value[0]);

		const imageURL = URL.createObjectURL(value[0]);

		setStoreImage(imageURL);

		const response = await fetch(fetchUrl, {
			method: 'POST',
			headers: {
				"accept": "application/json"
			},
			body: formData
		});

		const json = await response?.json();

		if (json) loading = false;

		setStoreResult(json);
	};
 
	const handleSubmit = (event) => {
		event.preventDefault();

		handlePostDetect();
	};
</script>

<form
	class="flex flex-col gap-3"
	on:submit={handleSubmit}
>
	<h1>Загрузите документ:</h1>
	<Uploader bind:value />
	{#if value.length > 0}
		<Button color="dark" pill type="submit" disabled={loading}>
			{`${loading ? 'Загрузка...' : 'Обработать'}`}
		</Button>
	{/if}
</form>
