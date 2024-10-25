<script>
  import { Dropzone, Button } from 'flowbite-svelte';
	import { DownloadSolid, CloseOutline } from 'flowbite-svelte-icons';

  export let value = [];

  const removeFile = (event) => {
    event.preventDefault();
    event.stopPropagation();

    value = [];
  };
	
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
  class="relative p-3"
  on:drop={dropHandle}
  on:dragover={(event) => {
    event.preventDefault();
  }}
  on:change={handleChange}
  disabled={value.length > 0}
  accept="image/png, image/jpeg, image/jpg"
>
	<DownloadSolid class="mb-3 w-12 h-12 text-gray-400" />
  {#if value.length === 0}
    <p class="mb-2 text-sm text-gray-500 dark:text-gray-400">
			<span class="font-semibold text-dark">Нажмите, чтобы загрузить</span>
			или перетащите
		</p>
    <p class="text-xs text-gray-500 dark:text-gray-400">
			Поддерживаемые форматы: PNG, JPG и JPEG
		</p>
  {:else}
    <p>{showFiles(value)}</p>
  {/if}

  {#if value.length > 0}
    <Button
      class="absolute top-2 right-2 p-0"
      on:click={removeFile}
    >
      <CloseOutline 
        class="w-5 h-5"
        color="gray"
      />
    </Button>
  {/if}
</Dropzone>