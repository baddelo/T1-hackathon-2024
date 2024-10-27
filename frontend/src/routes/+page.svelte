<script>
	import { Card } from 'flowbite-svelte';
	import { get } from 'svelte/store';

	import { resultStore } from '../store/result_store';

	import Form from '../lib/components/Form/index.svelte';
	import Result from '../lib/components/Result/index.svelte';
</script>

<Card class="p-5 min-h-80 max-w-xl">
	{#if $resultStore.length > 0}
		<Result />
	{:else}
		 <Form />
	{/if}
</Card>
