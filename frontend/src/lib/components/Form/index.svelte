<script>
	import { Button, Card } from 'flowbite-svelte';

	import Uploader from '../Uploader/index.svelte';
	
	let value = [];

	const handleSubmit = (event) => {
		event.preventDefault();
	};
</script>

<form
	class="flex flex-col gap-3"
	on:submit={handleSubmit}
>
	<h1>Загрузите документ:</h1>
	<Uploader bind:value />
	{#if value.length > 0}
		<Button color="dark" pill type="submit">
			Отправить
		</Button>
	{/if}
</form>