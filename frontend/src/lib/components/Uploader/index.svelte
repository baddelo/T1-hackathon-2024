<script>
  import { Dropzone } from 'flowbite-svelte';

	import { DownloadSolid } from 'flowbite-svelte-icons';

  let value = [];
	
  const dropHandle = (event) => {
    value = [];
    event.preventDefault();
    if (event.dataTransfer.items) {
      [...event.dataTransfer.items].forEach((item, i) => {
        if (item.kind === 'file') {
          const file = item.getAsFile();
          value.push(file.name);
          value = value;
        }
      });
    } else {
      [...event.dataTransfer.files].forEach((file, i) => {
        value = file.name;
      });
    }
  };

  const handleChange = (event) => {
    const files = event.target.files;
    if (files.length > 0) {
      value.push(files[0].name);
      value = value;
    }
  };

  const showFiles = (files) => {
    if (files.length === 1) return files[0];
    let concat = '';
    files.map((file) => {
      concat += file;
      concat += ',';
      concat += ' ';
    });

    if (concat.length > 40) concat = concat.slice(0, 40);
    concat += '...';
    return concat;
  };
</script>

<Dropzone
  id="dropzone"
  on:drop={dropHandle}
  on:dragover={(event) => {
    event.preventDefault();
  }}
  on:change={handleChange}
>
	<DownloadSolid class="mb-3 w-12 h-12 text-gray-400" />
  {#if value.length === 0}
    <p class="mb-2 text-sm text-gray-500 dark:text-gray-400">
			<span class="font-semibold text-blue-900 dark:text-blue-800">Нажмите, чтобы загрузить</span>
			или перетащите
		</p>
    <p class="text-xs text-gray-500 dark:text-gray-400">
			Поддерживаемые форматы: PNG, JPG и JPEG
		</p>
  {:else}
    <p>{showFiles(value)}</p>
  {/if}
</Dropzone>