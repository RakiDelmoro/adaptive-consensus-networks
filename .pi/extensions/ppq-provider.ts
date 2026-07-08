/**
 * PPQ.ai model provider extension
 *
 * Auth: Set PPQ_API_KEY in ~/.pi/agent/auth.json:
 *   { "ppq": { "type": "api_key", "key": "sk-..." } }
 * Or env var PPQ_API_KEY.
 */

// --- Type definitions ---
export type ProviderModelConfig = {
	id: string;
	name: string;
	api: string;
	reasoning: boolean;
	input: ("text" | "image")[];
	cost: {
		input: number;
		output: number;
		cacheRead: number;
		cacheWrite: number;
	};
	contextWindow: number;
};

export interface ExtensionAPI {
	registerProvider(
		name: string,
		config: {
			baseUrl: string;
			api: string;
			apiKey: string;
			models: ProviderModelConfig[];
			defaultModel?: string;
		},
	): void;
}

// --- inlined fp-sdk helpers (replaces OptionHelpers.ofObj / Some) ---
function ofObj<T>(val: T | null | undefined): T | null {
	return val != null ? val : null;
}

interface PPQPricing {
	input_per_1M_tokens: number;
	output_per_1M_tokens: number;
}

interface PPQArchitecture {
	modality: string;
	input_modalities: string[];
	output_modalities: string[];
}

interface PPQModel {
	id: string;
	name: string;
	context_length: number;
	pricing: PPQPricing;
	supported_parameters?: string[];
	architecture?: PPQArchitecture;
}

interface PPQApiResponse {
	data: PPQModel[];
}

const ppqApiBaseUrl = "https://api.ppq.ai";

function isMetaModel(modelId: string): boolean {
	const lowered = modelId.toLowerCase();
	return lowered.startsWith("auto") || lowered.startsWith("free");
}

async function fetchPpqModels(): Promise<PPQModel[]> {
	try {
			const response = await fetch(`${ppqApiBaseUrl}/v1/models`);
		if (!response.ok) {
		return [];
		}
		const data = (await response.json()) as PPQApiResponse;
		return data.data;
	} catch (error) {
		return [];
	}
}

async function filterPpqModels(apiModels: PPQModel[]): Promise<ProviderModelConfig[]> {
	try {
		const models: ProviderModelConfig[] = [];

		for (const model of apiModels) {
			const supportedParameters = ofObj(model.supported_parameters) ?? [];
			const architecture = ofObj(model.architecture);

			// pi requires models to have tool support
			if (!isMetaModel(model.id) && !supportedParameters.includes("tools")) {
				continue;
			}

			let inputModalities: ("text" | "image")[] = ["text"];
			if (architecture !== null) {
				inputModalities = architecture.input_modalities.filter(
					(m) => m === "text" || m === "image",
				);
			}
			models.push({
				id: model.id,
				name: model.name,
				api: "openai-completions",
				reasoning: supportedParameters.includes("reasoning"),
				input: inputModalities,
				cost: {
					input: model.pricing.input_per_1M_tokens,
					output: model.pricing.output_per_1M_tokens,
					cacheRead: 0,
					cacheWrite: 0,
				},
				contextWindow: model.context_length,
			} as ProviderModelConfig);
		}

		// Sort: autoclaw first, then auto, then alphabetical
		models.sort((a, b) => {
			const position = (id: string) => {
				if (id === "autoclaw") return 0;
				if (id === "auto") return 1;
				return 2;
			};
			const diff = position(a.id) - position(b.id);
			if (diff !== 0) return diff;
			return a.id.localeCompare(b.id);
		});

		return models;
	} catch (error) {
		return [];
	}
}

export default async function (pi: ExtensionAPI) {
	const apiModels = await fetchPpqModels();
	const models = await filterPpqModels(apiModels);
	if (models.length > 0) {
		// Set default model to the first one (autoclaw, auto, or first available)
		const defaultModel = models[0].id;
		pi.registerProvider("ppq", {
			baseUrl: ppqApiBaseUrl,
			api: "openai-completions",
			apiKey: "$PPQ_API_KEY",
			models: models,
			defaultModel: defaultModel,
		});

		// Show welcome message
		console.log('\n✓ PPQ.ai provider configured successfully!');
		console.log(`✓ ${models.length} models available, using '${defaultModel}' by default`);
		console.log('✓ To see all models: /models');
		console.log('✓ To switch models: /model <model-name>\n');
	} else {
		console.log('\n⚠ PPQ.ai provider: No models available. Please check your API key.\n');
	}
}
