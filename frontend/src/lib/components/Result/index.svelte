<script>
	import { get } from 'svelte/store';
	import { onMount, onDestroy } from 'svelte';
	import { Button } from 'flowbite-svelte';
	import { JsonView } from '@zerodevx/svelte-json-view';
	
	import {ArrowLeftOutline} from 'flowbite-svelte-icons';

	import {
		resultStore,
		resultImage,
		clearStoreData
	} from '../../../store/result_store.js';

	let image;
	let canvas;
	let context;

	let date = new Date().toLocaleDateString();
	let time = new Date().toLocaleTimeString();

	const drawRegions = () => {
		const coordinatesData = get(resultStore);

		if (!coordinatesData || coordinatesData.length === 0 || !image || !canvas) return;

		canvas.width = image.clientWidth;
		canvas.height = image.clientHeight;

		const scaleX = image.clientWidth / image.naturalWidth;
		const scaleY = image.clientHeight / image.naturalHeight;

		context.clearRect(0, 0, canvas.width, canvas.height);

		coordinatesData.forEach(region => {
			const [[x1, y1], [x2, y2]] = region.coordinates;

			const scaledX1 = x1 * scaleX;
			const scaledY1 = y1 * scaleY;
			const scaledX2 = x2 * scaleX;
			const scaledY2 = y2 * scaleY;

			const width = scaledX2 - scaledX1;
			const height = scaledY2 - scaledY1;

			context.beginPath();
			context.rect(scaledX1, scaledY1, width, height);
			context.strokeStyle = region.signature ? "red" : "blue";
			context.lineWidth = 1;
			context.stroke();

			if (region.content) {
				context.font = "10px Arial";
				context.fillStyle = "blue";
				context.fillText(region.content, scaledX1, scaledY1 - 5);
			}
		});
	};

	const handleResize = () => {
		drawRegions();
	};

	onMount(() => {
		image = document.getElementById("image");
		canvas = document.getElementById("canvas");
		context = canvas.getContext("2d");

		image.onload = drawRegions;

		window.addEventListener('resize', handleResize);

		return () => {
			window.removeEventListener('resize', handleResize);
		};
	});
</script>

<div class="flex flex-col gap-3 relative">
	<div class="flex flex-row gap-2">
		<button class="w-6 h-6 cursor-pointer" on:click={clearStoreData}>
			<ArrowLeftOutline class="w-full h-full" />
		</button>
		<h1>{date} {time}</h1>
	</div>
	<div style="position: relative; display: inline-block;">
		<img
			src={$resultImage}
			id="image"
			alt="resultImage"
			style="width: 100%; height: auto;"
		/>
		<canvas id="canvas" style="position: absolute; top: 0; left: 0;"></canvas>
	</div>
	<h2>Результат обработки:</h2>
	<div class="flex flex-col gap-1 h-60 overflow-x-auto border p-2 rounded-lg">
		{#each $resultStore as item}
			<p class="p-0">{item.content}</p>
		{/each}
	</div>
	<h2>JSON обработки:</h2>
	<div class="h-60 overflow-x-auto border p-2 rounded-lg">
		<JsonView json={$resultStore} />
	</div>
</div>
