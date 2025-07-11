from abc import abstractmethod, ABC
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Union, Optional, Callable, Any, Dict, TypeVar, Literal


class Environment(Enum):
    PRODUCTION = "PRODUCTION"
    DEVELOPMENT = "DEVELOPMENT"
    TEST = "TEST"


class EPromptWeighting(Enum):
    COMPEL = "compel"
    SDEMBEDS = "sdembeds"


class SdkType(Enum):
    CLIENT = "CLIENT"
    SERVER = "SERVER"


class EControlMode(Enum):
    BALANCED = "balanced"
    PROMPT = "prompt"
    CONTROL_NET = "controlnet"


class ETaskType(Enum):
    IMAGE_INFERENCE = "imageInference"
    PHOTO_MAKER = "photoMaker"
    IMAGE_UPLOAD = "imageUpload"
    IMAGE_UPSCALE = "imageUpscale"
    IMAGE_BACKGROUND_REMOVAL = "imageBackgroundRemoval"
    IMAGE_CAPTION = "imageCaption"
    IMAGE_CONTROL_NET_PRE_PROCESS = "imageControlNetPreProcess"
    PROMPT_ENHANCE = "promptEnhance"
    AUTHENTICATION = "authentication"
    MODEL_UPLOAD = "modelUpload"
    MODEL_SEARCH = "modelSearch"
    VIDEO_INFERENCE = "videoInference"
    GET_RESPONSE = "getResponse"


class EPreProcessorGroup(Enum):
    canny = "canny"
    depth = "depth"
    mlsd = "mlsd"
    normalbae = "normalbae"
    openpose = "openpose"
    tile = "tile"
    seg = "seg"
    lineart = "lineart"
    lineart_anime = "lineart_anime"
    shuffle = "shuffle"
    scribble = "scribble"
    softedge = "softedge"


class EPreProcessor(Enum):
    canny = "canny"
    depth_leres = "depth_leres"
    depth_midas = "depth_midas"
    depth_zoe = "depth_zoe"
    inpaint_global_harmonious = "inpaint_global_harmonious"
    lineart_anime = "lineart_anime"
    lineart_coarse = "lineart_coarse"
    lineart_realistic = "lineart_realistic"
    lineart_standard = "lineart_standard"
    mlsd = "mlsd"
    normal_bae = "normal_bae"
    scribble_hed = "scribble_hed"
    scribble_pidinet = "scribble_pidinet"
    seg_ofade20k = "seg_ofade20k"
    seg_ofcoco = "seg_ofcoco"
    seg_ufade20k = "seg_ufade20k"
    shuffle = "shuffle"
    softedge_hed = "softedge_hed"
    softedge_hedsafe = "softedge_hedsafe"
    softedge_pidinet = "softedge_pidinet"
    softedge_pidisafe = "softedge_pidisafe"
    tile_gaussian = "tile_gaussian"
    openpose = "openpose"
    openpose_face = "openpose_face"
    openpose_faceonly = "openpose_faceonly"
    openpose_full = "openpose_full"
    openpose_hand = "openpose_hand"


class EOpenPosePreProcessor(Enum):
    openpose = "openpose"
    openpose_face = "openpose_face"
    openpose_faceonly = "openpose_faceonly"
    openpose_full = "openpose_full"
    openpose_hand = "openpose_hand"


# Define the types using Literal
IOutputType = Literal["base64Data", "dataURI", "URL"]
IOutputFormat = Literal["JPG", "PNG", "WEBP"]


@dataclass
class File:
    data: bytes


@dataclass
class RunwareBaseType:
    apiKey: str
    url: Optional[str] = None


@dataclass
class IImage:
    taskType: str
    imageUUID: str
    taskUUID: str
    seed: Optional[int] = None
    inputImageUUID: Optional[str] = None
    imageURL: Optional[str] = None
    imageBase64Data: Optional[str] = None
    imageDataURI: Optional[str] = None
    NSFWContent: Optional[bool] = None
    cost: Optional[float] = None


@dataclass
class ILora:
    model: str
    weight: float


@dataclass
class ILycoris:
    model: str
    weight: float


@dataclass
class IEmbedding:
    model: str


@dataclass
class IRefiner:
    model: Union[int, str]
    startStep: Optional[int] = None
    startStepPercentage: Optional[float] = None


@dataclass(kw_only=True)
class IControlNetGeneral:
    model: str
    guideImage: Union[str, File]
    weight: Optional[float] = None
    startStep: Optional[int] = None
    endStep: Optional[int] = None
    startStepPercentage: Optional[int] = None
    endStepPercentage: Optional[int] = None
    controlMode: Optional[EControlMode] = None

    def __post_init__(self):
        if (self.startStep and self.startStepPercentage) or (
            self.endStep and self.endStepPercentage
        ):
            raise ValueError(
                "Exactly one of 'startStep/endStep' or 'startStepPercentage/endStepPercentage' must be provided."
            )


@dataclass
class IControlNetCanny(IControlNetGeneral):
    lowThresholdCanny: Optional[int] = None
    highThresholdCanny: Optional[int] = None
    preprocessor: EPreProcessor = EPreProcessor.canny


@dataclass
class IControlNetOpenPose(IControlNetGeneral):
    model: Optional[str] = None
    includeHandsAndFaceOpenPose: bool = True
    preprocessor: EOpenPosePreProcessor = EOpenPosePreProcessor.openpose


IControlNet = Union[IControlNetGeneral, IControlNetCanny, IControlNetOpenPose]


@dataclass
class IError:
    error: bool
    error_message: str
    task_uuid: str
    error_code: Optional[str] = None
    parameter: Optional[str] = None
    error_type: Optional[str] = None
    documentation: Optional[str] = None


class EModelArchitecture(Enum):
    FLUX1D = "flux1d"
    FLUX1S = "flux1s"
    PONY = "pony"
    SDHYPER = "sdhyper"
    SD1X = "sd1x"
    SD1XLCM = "sd1xlcm"
    SD3 = "sd3"
    SDXL = "sdxl"
    SDXL_LCM = "sdxllcm"
    SDXL_DISTILLED = "sdxldistilled"
    SDXL_HYPER = "sdxlhyper"
    SDXL_LIGHTNING = "sdxllightning"
    SDXL_TURBO = "sdxlturbo"


@dataclass
class IModel:
    air: str
    name: str
    version: str
    category: str
    architecture: str
    tags: List[str]
    heroImage: str
    private: bool
    comment: str

    type: Optional[str] = None
    defaultWidth: Optional[int] = None
    defaultHeight: Optional[int] = None
    defaultSteps: Optional[int] = None
    defaultScheduler: Optional[str] = None
    defaultCFG: Optional[float] = None
    defaultStrength: float = 0.0
    conditioning: Optional[str] = None
    positiveTriggerWords: Optional[str] = None

    additional_fields: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for key, value in self.additional_fields.items():
            setattr(self, key, value)


@dataclass
class IModelSearchResponse:
    results: List[IModel]
    taskUUID: str
    taskType: str
    totalResults: int


@dataclass
class IModelSearch:
    search: Optional[str] = None
    tags: Optional[List[str]] = None
    category: Optional[Literal["checkpoint", "lora", "controlnet"]] = None
    type: Optional[str] = None
    architecture: Optional[EModelArchitecture] = None
    conditioning: Optional[str] = None
    visibility: Optional[Literal["public", "private", "all"]] = None
    limit: int = 20
    offset: int = 0
    customTaskUUID: Optional[str] = None
    retry: Optional[int] = None
    additional_params: Dict[str, Union[str, int, float, bool, None]] = field(
        default_factory=dict
    )

    def __post_init__(self):
        standard_fields = {
            "search",
            "tags",
            "category",
            "type",
            "architecture",
            "conditioning",
            "visibility",
            "limit",
            "offset",
            "customTaskUUID",
            "retry",
        }
        for key in list(self.additional_params.keys()):
            if key in standard_fields:
                del self.additional_params[key]


@dataclass
class IPhotoMaker:
    model: Union[int, str]
    positivePrompt: str
    height: int
    width: int
    numberResults: int = 1
    steps: Optional[int] = None
    outputType: Optional[IOutputType] = None
    inputImages: List[Union[str, File]] = field(default_factory=list)
    style: Optional[str] = None
    strength: Optional[float] = None
    outputFormat: Optional[IOutputFormat] = None
    includeCost: Optional[bool] = None
    taskUUID: Optional[str] = None

    def __post_init__(self):
        # Validate `inputImages` to ensure it has a maximum of 4 elements
        if len(self.inputImages) > 4:
            raise ValueError("inputImages can contain a maximum of 4 elements.")

        # Validate `style` to ensure it matches one of the allowed case-sensitive options
        valid_styles = {
            "No style",
            "Cinematic",
            "Disney Character",
            "Digital Art",
            "Photographic",
            "Fantasy art",
            "Neonpunk",
            "Enhance",
            "Comic book",
            "Lowpoly",
            "Line art",
        }
        if self.style and self.style not in valid_styles:
            raise ValueError(
                f"style must be one of the following: {', '.join(valid_styles)}."
            )


@dataclass
class IOutpaint:
    top: Optional[int] = None
    right: Optional[int] = None
    bottom: Optional[int] = None
    left: Optional[int] = None
    blur: Optional[int] = None


@dataclass
class IInstantID:
    inputImage: Union[File, str]
    poseImage: Optional[Union[File, str]] = None
    identityNetStrength: Optional[float] = None
    adapterStrength: Optional[float] = None
    controlNetCannyWeight: Optional[float] = None
    controlNetDepthWeight: Optional[float] = None
    enhanceNonFaceRegion: bool = True


@dataclass
class IIpAdapter:
    model: Union[int, str]
    guideImage: Union[File, str]
    weight: Optional[float] = None


@dataclass
class IAcePlusPlus:
    taskType: str
    repaintingScale: float = 0.0
    inputImages: Optional[List[Union[str, File]]] = field(default_factory=list)
    inputMasks: Optional[List[Union[str, File]]] = field(default_factory=list)
    _VALID_TASK_TYPES = ("portrait", "subject", "local_editing")

    def __post_init__(self):
        # Validate repaintingScale
        if not 0.0 <= self.repaintingScale <= 1.0:
            raise ValueError("repaintingScale must be between 0.0 and 1.0")

        # Validate taskType
        if self.taskType not in self._VALID_TASK_TYPES:
            raise ValueError(
                f"taskType must be one of {self._VALID_TASK_TYPES}, got: {self.taskType}"
            )


@dataclass
class IAcceleratorOptions:
    teaCache: Optional[bool] = None
    cacheStartStep: Optional[int] = None
    cacheStopStep: Optional[int] = None
    teaCacheDistance: Optional[float] = None
    deepCache: Optional[bool] = None
    deepCacheInterval: Optional[float] = None
    deepCacheBranchId: Optional[int] = None
    deepCacheSkipMode: Optional[str] = None


@dataclass
class IFluxKontext:
    guidanceEndStep: Optional[int] = None
    guidanceEndStepPercentage: Optional[float] = None


@dataclass
class IAdvancedFeatures:
    fluxKontext: Optional[IFluxKontext] = None


@dataclass
class IImageInference:
    """
    Represents the parameters for an image inference task.

    Attributes:
        positivePrompt (str):
            Required. A positive prompt is a text instruction to guide the model on generating the image. The length must be between 2 and 3000 characters. Use '__BLANK__' for no prompt guidance.
        model (Union[int, str]):
            Required. The AIR identifier of the model to use for inference.
        taskUUID (Optional[str]):
            Required. UUID v4. A unique identifier for the task, used to match async responses to their corresponding tasks. Must be unique for each task.
        outputType (Optional[IOutputType]):
            Specifies the output type for the image: 'base64Data', 'dataURI', or 'URL'. Default is 'URL'.
        outputFormat (Optional[IOutputFormat]):
            Specifies the format of the output image: 'JPG', 'PNG', or 'WEBP'. Default is 'JPG'.
        uploadEndpoint (Optional[str]):
            URL to which the generated image will be uploaded as binary data using HTTP PUT.
        checkNsfw (Optional[bool]):
            Enables or disables NSFW content check. Default is False. Adds 0.1s to inference time and may incur additional costs.
        negativePrompt (Optional[str]):
            A negative prompt to guide the model on what to avoid in the image. Length must be between 2 and 3000 characters.
        seedImage (Optional[Union[File, str]]):
            Required for image-to-image, inpainting, or outpainting. Specifies the seed image in UUID, data URI, base64, or URL format.
        referenceImages (Optional[Union[File, str]]):
            An array of reference images to condition the generation process. Useful for edit models and ACE++ workflows.
        maskImage (Optional[Union[File, str]]):
            Required for inpainting. Specifies the mask image in UUID, data URI, base64, or URL format.
        strength (Optional[float]):
            For image-to-image or inpainting. Value between 0 and 1. Controls the influence of the seed image. Default is 0.8.
        height (Optional[int]):
            Required. Height of the generated image (128-2048, divisible by 64).
        width (Optional[int]):
            Required. Width of the generated image (128-2048, divisible by 64).
        acceleratorOptions (Optional[IAcceleratorOptions]):
            Advanced caching mechanisms to speed up image generation.
        advancedFeatures (Optional[IAdvancedFeatures]):
            Specialized features that extend the image generation process, such as LayerDiffuse for transparency.
        steps (Optional[int]):
            Number of inference steps (1-100). Default is 20.
        scheduler (Optional[str]):
            Scheduler to use for inference. Default is the model's scheduler.
        seed (Optional[int]):
            Seed for randomization. If set, will be incremented for each image generated.
        CFGScale (Optional[float]):
            Guidance scale (0-50). Default is 7. Higher values adhere more closely to the prompt.
        clipSkip (Optional[int]):
            Additional layer skips during prompt processing in the CLIP model (0-2).
        promptWeighting (Optional[EPromptWeighting]):
            Syntax to use for prompt weighting (e.g., 'compel', 'sdembeds').
        numberResults (Optional[int]):
            Number of images to generate (1-20). Default is 1.
        controlNet (Optional[List[IControlNet]]):
            List of ControlNet configurations for guided image generation.
        lora (Optional[List[ILora]]):
            List of LoRA models for style or feature adaptation.
        lycoris (Optional[List[ILycoris]]):
            List of LyCORIS models for additional adaptation.
        includeCost (Optional[bool]):
            If True, the response will include the cost of the task. Default is False.
        onPartialImages (Optional[Callable[[List[IImage], Optional[IError]], None]]):
            Callback for handling partial image results during async processing.
        refiner (Optional[IRefiner]):
            Refiner model configuration for enhanced image quality (SDXL-based only).
        vae (Optional[str]):
            AIR identifier for a VAE model to override the default.
        maskMargin (Optional[int]):
            Adds extra context pixels (32-128) around the masked region during inpainting.
        outputQuality (Optional[int]):
            Compression quality of the output image (20-99). Default is 95.
        embeddings (Optional[List[IEmbedding]]):
            List of embedding models (Textual Inversion) to add specific concepts or styles.
        outpaint (Optional[IOutpaint]):
            Extends image boundaries in specified directions. Requires width and height to account for the extension.
        instantID (Optional[IInstantID]):
            PuLID configuration for fast and high-quality identity customization.
        ipAdapters (Optional[List[IIpAdapter]]):
            List of IP-Adapter models for image-prompted generation.
        referenceImages (Optional[List[Union[str, File]]]):
            Reference images for conditioning (used in some workflows, e.g., ACE++).
        acePlusPlus (Optional[IAcePlusPlus]):
            ACE++ configuration for character-consistent image generation and editing.
        extraArgs (Optional[Dict[str, Any]]):
            Additional arguments for extensibility.
    """
    positivePrompt: str
    model: Union[int, str]
    taskUUID: Optional[str] = None
    outputType: Optional[IOutputType] = None
    outputFormat: Optional[IOutputFormat] = None
    uploadEndpoint: Optional[str] = None
    checkNsfw: Optional[bool] = None
    negativePrompt: Optional[str] = None
    seedImage: Optional[Union[File, str]] = None
    referenceImages: Optional[Union[File, str]] = None
    maskImage: Optional[Union[File, str]] = None
    strength: Optional[float] = None
    height: Optional[int] = None
    width: Optional[int] = None
    acceleratorOptions: Optional[IAcceleratorOptions] = None
    advancedFeatures: Optional[IAdvancedFeatures] = None
    steps: Optional[int] = None
    scheduler: Optional[str] = None
    seed: Optional[int] = None
    CFGScale: Optional[float] = None
    clipSkip: Optional[int] = None
    promptWeighting: Optional[EPromptWeighting] = None
    numberResults: Optional[int] = 1  # default to 1
    controlNet: Optional[List[IControlNet]] = field(default_factory=list)
    lora: Optional[List[ILora]] = field(default_factory=list)
    lycoris: Optional[List[ILycoris]] = field(default_factory=list)
    includeCost: Optional[bool] = None
    onPartialImages: Optional[Callable[[List[IImage], Optional[IError]], None]] = None
    refiner: Optional[IRefiner] = None
    vae: Optional[str] = None
    maskMargin: Optional[int] = None
    outputQuality: Optional[int] = None
    embeddings: Optional[List[IEmbedding]] = field(default_factory=list)
    outpaint: Optional[IOutpaint] = None
    instantID: Optional[IInstantID] = None
    ipAdapters: Optional[List[IIpAdapter]] = field(default_factory=list)
    referenceImages: Optional[List[Union[str, File]]] = field(default_factory=list)
    acePlusPlus: Optional[IAcePlusPlus] = None
    extraArgs: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class IImageCaption:
    """
    Represents the parameters for an image caption task.

    Attributes:
        taskType (str):
            Required. The type of task to be performed. For this task, the value should be 'imageCaption'.
        taskUUID (str):
            Required. UUID v4. When a task is sent to the API you must include a random UUID v4 string using the taskUUID parameter. This string is used to match the async responses to their corresponding tasks. If you send multiple tasks at the same time, the taskUUID will help you match the responses to the correct tasks. The taskUUID must be unique for each task you send to the API.
        inputImage (Optional[Union[File, str]]):
            Required. Specifies the input image to be processed. The image can be specified in one of the following formats:
            - An UUID v4 string of a previously uploaded image or a generated image.
            - A data URI string representing the image. The data URI must be in the format data:<mediaType>;base64, followed by the base64-encoded image. For example: data:image/png;base64,iVBORw0KGgo....
            - A base64 encoded image without the data URI prefix. For example: iVBORw0KGgo....
            - A URL pointing to the image. The image must be accessible publicly.
            Supported formats are: PNG, JPG and WEBP.
        includeCost (bool):
            If set to true, the cost to perform the task will be included in the response object. Default is False.
    """

    inputImage: Optional[Union[File, str]] = None
    includeCost: bool = False


@dataclass
class IImageToText:
    taskType: ETaskType
    taskUUID: str
    text: str
    cost: Optional[float] = None


@dataclass
class IBackgroundRemovalSettings:
    returnOnlyMask: bool = False
    alphaMatting: bool = False
    postProcessMask: bool = False
    alphaMattingErodeSize: Optional[int] = None
    alphaMattingForegroundThreshold: Optional[int] = None
    alphaMattingBackgroundThreshold: Optional[int] = None
    rgba: Optional[List[int]] = None


@dataclass
class IImageBackgroundRemoval(IImageCaption):
    """
    Parameters for an image background removal task.

    Attributes:
        outputType (Optional[IOutputType]):
            Specifies the output type in which the image is returned. Supported values: 'base64Data', 'dataURI', 'URL'. Default is 'URL'.
            - base64Data: The image is returned as a base64-encoded string (imageBase64Data).
            - dataURI: The image is returned as a data URI string (imageDataURI).
            - URL: The image is returned as a URL string (imageURL).
        outputFormat (Optional[IOutputFormat]):
            Specifies the format of the output image. Supported formats: 'PNG', 'JPG', 'WEBP'. Default is 'PNG'.
        outputQuality (Optional[int]):
            Sets the compression quality of the output image (20-99). Default is 95. Higher values preserve more quality but increase file size.
        model (Optional[Union[int, str]]):
            The AIR identifier of the background removal model to use. This is a unique string representing a specific model.
        taskUUID (Optional[str]):
            UUID v4. A unique identifier for the task, used to match async responses to their corresponding tasks.
        settings (Optional[IBackgroundRemovalSettings]):
            An object containing all background removal configuration options. Currently, only supported by RemBG 1.4 (runware:109@1).
    """
    outputType: Optional[IOutputType] = None
    outputFormat: Optional[IOutputFormat] = None
    outputQuality: Optional[int] = None
    model: Optional[Union[int, str]] = None
    taskUUID: Optional[str] = None
    settings: Optional[IBackgroundRemovalSettings] = None


@dataclass
class IPromptEnhance:
    """
    Represents the parameters for a prompt enhancement task.

    Attributes:
        promptMaxLength (int):
            Required. The maximum length of the enhanced prompt to receive, expressed in tokens (12-400). Approximately 100 tokens correspond to about 75 words or 500 characters.
        promptVersions (int):
            Required. The number of prompt versions to receive (1-5).
        prompt (str):
            Required. The prompt to enhance (1-300 characters).
        includeCost (bool):
            If set to True, the cost to perform the task will be included in the response object. Default is False.
    """
    promptMaxLength: int
    promptVersions: int
    prompt: str
    includeCost: bool = False


@dataclass
class IEnhancedPrompt(IImageToText):
    pass

    def __hash__(self):
        return hash((self.taskType, self.taskUUID, self.text, self.cost))


@dataclass
class IImageUpscale:
    inputImage: Union[str, File]
    upscaleFactor: int
    outputType: Optional[IOutputType] = None
    outputFormat: Optional[IOutputFormat] = None
    includeCost: bool = False


class ReconnectingWebsocketProps:
    def __init__(self, websocket: Any):
        self.websocket = websocket

    def add_event_listener(self, event_type: str, listener: Callable, options: Any):
        self.websocket.addEventListener(event_type, listener, options)

    def send(self, data: Any):
        self.websocket.send(data)

    def __getattr__(self, name: str):
        return getattr(self.websocket, name)


@dataclass
class UploadImageType:
    imageUUID: str
    imageURL: str
    taskUUID: str


@dataclass
class IUploadModelBaseType:
    air: str
    architecture: str
    name: str
    downloadURL: str
    uniqueIdentifier: str
    version: str
    format: str
    private: bool
    category: str
    heroImageURL: Optional[str] = None
    tags: Optional[List[str]] = field(default_factory=list)
    shortDescription: Optional[str] = None
    comment: Optional[str] = None
    retry: Optional[int] = None


@dataclass
class IUploadModelControlNet(IUploadModelBaseType):
    category: str = "controlnet"
    conditioning: Optional[str] = None

    def __post_init__(self):
        if self.conditioning is None:
            raise ValueError("conditioning is required for IUploadModelCheckPoint")


@dataclass
class IUploadModelCheckPoint(IUploadModelBaseType):
    category: str = "checkpoint"
    defaultScheduler: Optional[str] = None
    type: Optional[str] = None
    defaultStrength: Optional[float] = None
    defaultWeight: Optional[float] = None
    positiveTriggerWords: Optional[str] = None
    defaultGuidanceScale: Optional[float] = None
    defaultSteps: Optional[int] = None
    negativeTriggerWords: Optional[str] = None

    def __post_init__(self):
        if self.type is None:
            raise ValueError("type is required for IUploadModelCheckPoint")

        if self.defaultScheduler is None:
            raise ValueError("defaultScheduler is required for IUploadModelCheckPoint")


@dataclass
class IUploadModelLora(IUploadModelBaseType):
    category: str = "lora"
    defaultWeight: Optional[float] = None
    positiveTriggerWords: Optional[str] = None


@dataclass
class IUploadModelResponse:
    air: str
    taskUUID: str
    taskType: str


@dataclass
class IFrameImage:
    inputImage: Union[str, File]
    frame: Optional[Union[Literal["first", "last"], int]] = None


class SerializableMixin:
    def serialize(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items()
                if v is not None and not k.startswith('_')}


@dataclass
class BaseProviderSettings(SerializableMixin, ABC):
    @property
    @abstractmethod
    def provider_key(self) -> str:
        pass

    def to_request_dict(self) -> Dict[str, Any]:
        data = self.serialize()
        if data:
            return {self.provider_key: data}
        return {}


@dataclass
class IKlingCameraConfig(SerializableMixin):
    horizontal: Optional[int] = None
    vertical: Optional[int] = None
    zoom: Optional[int] = None
    roll: Optional[int] = None
    tilt: Optional[int] = None
    pan: Optional[int] = None


@dataclass
class IKlingCameraControl(SerializableMixin):
    camera_type: Optional[str] = None
    config: Optional[IKlingCameraConfig] = None

    def serialize(self) -> Dict[str, Any]:
        result = {}
        if self.camera_type:
            result["type"] = self.camera_type
        if self.config:
            config_data = self.config.serialize()
            if config_data:
                result["config"] = config_data
        return result


@dataclass
class IGoogleProviderSettings(BaseProviderSettings):
    generateAudio: Optional[bool] = None
    enhancePrompt: Optional[bool] = None

    @property
    def provider_key(self) -> str:
        return "google"


@dataclass
class IMinimaxProviderSettings(BaseProviderSettings):
    promptOptimizer: Optional[bool] = None

    @property
    def provider_key(self) -> str:
        return "minimax"


@dataclass
class IBytedanceProviderSettings(BaseProviderSettings):
    cameraFixed: Optional[bool] = None

    @property
    def provider_key(self) -> str:
        return "bytedance"


@dataclass
class IKlingAIProviderSettings(BaseProviderSettings):
    cameraControl: Optional[IKlingCameraControl] = None

    @property
    def provider_key(self) -> str:
        return "klingai"

    def serialize(self) -> Dict[str, Any]:
        result = {}
        if self.cameraControl:
            camera_control_data = self.cameraControl.serialize()
            if camera_control_data:
                result["cameraControl"] = camera_control_data
        return result


VideoProviderSettings = IKlingAIProviderSettings | IGoogleProviderSettings | IMinimaxProviderSettings | IBytedanceProviderSettings

@dataclass
class IVideoInference:
    model: str
    positivePrompt: str
    duration: float | None = None
    width: int | None = None
    height: int | None = None
    deliveryMethod: str = "async"
    taskUUID: Optional[str] = None
    outputType: Optional[IOutputType] = None
    outputFormat: Optional[Literal["MP4", "WEBM"]] = None
    outputQuality: Optional[int] = None
    uploadEndpoint: Optional[str] = None
    includeCost: Optional[bool] = None
    negativePrompt: Optional[str] = None
    frameImages: Optional[List[Union[IFrameImage, str]]] = field(default_factory=list)
    referenceImages: Optional[List[Union[str, File]]] = field(default_factory=list)
    fps: Optional[int] = None
    steps: Optional[int] = None
    seed: Optional[int] = None
    CFGScale: Optional[float] = None
    numberResults: Optional[int] = 1
    providerSettings: Optional[VideoProviderSettings] = None

@dataclass
class IVideo:
    taskType: str
    taskUUID: str
    status: Optional[str] = None
    videoUUID: Optional[str] = None
    videoURL: Optional[str] = None
    cost: Optional[float] = None
    seed: Optional[int] = None


# The GetWithPromiseCallBackType is defined using the Callable type from the typing module. It represents a function that takes a dictionary
# with specific keys and returns either a boolean or None.
# The dictionary should have the following keys:
# resolve: A function that takes a value of any type and returns None.
# reject: A function that takes a value of any type and returns None.
# intervalId: A value of any type representing the interval ID.
# You can use these types in your Python code to define variables, parameters, or return types that match the corresponding TypeScript types.
#
# def on_message(event: Any):
#     # Handle WebSocket message event
#     pass
#
# websocket = ReconnectingWebsocketProps(websocket_object)
# websocket.add_event_listener("message", on_message, {})
#
# uploaded_image = UploadImageType("abc123", "image.png", "task123")
#
# def get_with_promise(callback_data: Dict[str, Union[Callable[[Any], None], Any]]) -> Union[bool, None]:
#     # Implement the callback function logic here
#     pass


GetWithPromiseCallBackType = Callable[
    [Dict[str, Union[Callable[[Any], None], Any]]], Union[bool, None]
]


# The ListenerType class is defined to represent the structure of a listener.
# The key parameter is a string that represents a unique identifier for the listener.
# The listener parameter is a callable function that takes a single argument msg of type Any and returns None.
# It represents the function to be called when the corresponding event occurs.
# The group_key parameter is an optional string that represents a group identifier for the listener. It allows grouping listeners together based on a common key.
# You can create instances of ListenerType by providing the required parameters:
#
# def on_message(msg: Any):
#     # Handle the message
#     print(msg)
#
# listener = ListenerType("message_listener", on_message, group_key="message_group")

# In this example, we define a function on_message that takes a single argument msg and handles the received message.
# We then create an instance of ListenerType called listener by providing the key "message_listener",
# the on_message function as the listener, and an optional group key "message_group".
# You can store instances of ListenerType in a list or dictionary to manage multiple listeners in your application.

# listeners = [
#     ListenerType("listener1", on_message1),
#     ListenerType("listener2", on_message2, group_key="group1"),
#     ListenerType("listener3", on_message3, group_key="group1"),
# ]


class ListenerType:
    def __init__(
        self,
        key: str,
        listener: Callable[[Any], None],
        group_key: Optional[str] = None,
        debug_message: Optional[str] = None,
    ):
        """
        Initialize a new ListenerType instance.

        :param key: str, a unique identifier for the listener.
        :param listener: Callable[[Any], None], the function to be called when the listener is triggered.
        :param group_key: Optional[str], an optional grouping key that can be used to categorize listeners.
        """
        self.key = key
        self.listener = listener
        self.group_key = group_key
        self.debug_message = debug_message

    def __str__(self):
        return f"ListenerType(key={self.key}, listener={self.listener}, group_key={self.group_key}, debug_message={self.debug_message})"

    def __repr__(self):
        return self.__str__()


T = TypeVar("T")
Keys = TypeVar("Keys")


class RequireAtLeastOne:
    def __init__(self, data: Dict[str, Any], required_keys: Union[str, Keys]):
        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary")

        self.data = data
        self.required_keys = required_keys

        if not isinstance(required_keys, (list, tuple)):
            required_keys = [required_keys]

        missing_keys = [key for key in required_keys if key not in data]
        if len(missing_keys) == len(required_keys):
            raise ValueError(
                f"At least one of the required keys must be present: {', '.join(required_keys)}"
            )

    def __getitem__(self, key: str):
        return self.data[key]

    def __setitem__(self, key: str, value: Any):
        self.data[key] = value

    def __delitem__(self, key: str):
        del self.data[key]

    def __contains__(self, key: str):
        return key in self.data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)


class RequireOnlyOne(RequireAtLeastOne):
    def __init__(self, data: Dict[str, Any], required_keys: Union[str, Keys]):
        super().__init__(data, required_keys)

        if not isinstance(required_keys, (list, tuple)):
            required_keys = [required_keys]

        provided_keys = [key for key in required_keys if key in data]
        if len(provided_keys) > 1:
            raise ValueError(
                f"Only one key can be provided: {', '.join(provided_keys)}"
            )
