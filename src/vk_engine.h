#pragma once

#include "vk_types.h"
#include "vk_descriptors.h"
#include "vk_camera.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <VkBootstrap.h>

#include <functional>


// from: https://stackoverflow.com/a/57595105
template <typename T, typename... Rest>
void hashCombine(std::size_t& seed, const T& v, const Rest&... rest) {
	seed ^= std::hash<T>{}(v)+0x9e3779b9 + (seed << 6) + (seed >> 2);
	(hashCombine(seed, rest), ...);
};

//------OG VULKAN TUTO-------//
struct UniformBufferObject {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
	glm::mat4 inverseView;
};

struct Vertex {
	glm::vec3 position;
	glm::vec3 color;
	glm::vec3 normal;
	glm::vec2 uv;

	static VkVertexInputBindingDescription getBindingDescription() {
		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescription;
	}

	static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions() {
		std::vector<VkVertexInputAttributeDescription> attributeDescriptions{};

		attributeDescriptions.push_back({ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) });
		attributeDescriptions.push_back({ 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) });
		attributeDescriptions.push_back({ 2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) });
		attributeDescriptions.push_back({ 3, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv) });

		return attributeDescriptions;
	}

	bool operator==(const Vertex& other) const {
		return position == other.position && color == other.color && normal == other.normal &&
			uv == other.uv;
	}
};

struct DeletionQueue
{
	std::deque<std::function<void()>> deletors;

	void push_function(std::function<void()>&& function) {
		deletors.push_back(function);
	}

	void flush() {
		// reverse iterate the deletion queue to execute all the functions
		for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
			(*it)(); //call functors
		}

		deletors.clear();
	}
};


struct FrameData {
	VkSemaphore _swapchainSemaphore, _renderSemaphore;
	VkFence _renderFence;
	VkCommandBuffer _mainCommandBuffer;
	DeletionQueue _deletionQueue;
};

constexpr unsigned int FRAME_OVERLAP = 2;


class VulkanEngine {
public:
	GLFWwindow* _window;
	DeletionQueue _mainDeletionQueue;
	
	VmaAllocator _allocator;
	VkCommandPool _commandPool;
	//draw resources
	AllocatedImage _drawImage;

	VkInstance _instance;// Vulkan library handle
	VkDebugUtilsMessengerEXT _debug_messenger;// Vulkan debug output handle
	VkPhysicalDevice _chosenGPU;// GPU chosen as the default device
	VkDevice _device; // Vulkan device for commands
	VkSurfaceKHR _surface;// Vulkan window surface

	VkSwapchainKHR _swapchain;
	VkFormat _swapchainImageFormat;
	VkExtent2D _swapchainExtent;
	std::vector<VkImage> _swapchainImages;
	std::vector<VkImageView> _swapchainImageViews;

	VkExtent2D _windowExtent{ 800 , 600 };

	VkQueue _graphicsQueue;
	uint32_t _graphicsQueueFamily;


	FrameData _frames[FRAME_OVERLAP];

	FrameData& get_current_frame() { return _frames[_frameNumber % FRAME_OVERLAP]; };

	bool _isInitialized{ false };
	int _frameNumber{ 0 };
	bool stop_rendering{ false };

	static VulkanEngine& Get();

	//initializes everything in the engine
	void init();

	//shuts down the engine
	void cleanup();

	//draw loop
	void draw();

	//run main loop
	void run();


private:
	void init_vulkan();
	void init_swapchain();
	void init_commands();
	void init_sync_structures();
	void create_swapchain(uint32_t width, uint32_t height);
	void destroy_swapchain();
	void draw_background(VkCommandBuffer cmd);

	MyCamera _camera;

	void createStencilPipeline(const std::string& vertShaderPath, const std::string& fragShaderPath, VkPipeline& pipeline, VkPipelineLayout& pipelineLayout);
	VkPipeline stencilPipeline;
	VkPipelineLayout stencilPipelineLayout;

	void createForwardPipeline(const std::string& vertShaderPath, const std::string& fragShaderPath, VkPipeline& pipeline, VkPipelineLayout& pipelineLayout);
	VkPipeline forwardPipeline;
	VkPipelineLayout forwardPipelineLayout;
	VkPipeline forwardPipelineLight;
	VkPipelineLayout forwardPipelineLightLayout;

	void createPointLightPipeline(const std::string& vertShaderPath, const std::string& fragShaderPath, VkPipeline& pipeline, VkPipelineLayout& pipelineLayout);
	VkPipeline pointLightPipeline;
	VkPipelineLayout pointLightPipelineLayout;

	//>------for DEFERRED-------//
	AllocatedImage _gPosition;
	AllocatedImage _gNormal;
	AllocatedImage _gAlbedoSpec;

	DescriptorAllocator globalDescriptorAllocator;
	VkDescriptorSet _drawImageDescriptorSet;
	

	VkSampler _defaultSamplerLinear;
	VkSampler _defaultSamplerNearest;

	VkPipelineLayout _secondPassPipelineLayout;
	VkPipeline _secondPassPipeline;

	void init_images();
	void init_descriptors();
	void init_secondPass_pipeline();

	//<-------for DEFERRED------//
	//>old pipeline
	void createGPipeline(const std::string& vertShaderPath, const std::string& fragShaderPath, VkPipeline& pipeline, VkPipelineLayout& pipelineLayout);
	VkShaderModule createShaderModule(const std::vector<char>& code);
	static std::vector<char> readFile(const std::string& filename); //for shader
	AllocatedImage _depthImage;
	VkImageView _stencilView;
	AllocatedImage _secondPassImage;

	VkPipeline graphicsPipeline;
	VkPipeline gPipeline;
	VkPipelineLayout pipelineLayout;
	VkPipelineLayout gPipelineLayout;

	//<old pipeline

	//>----------LVE VULKAN TUTO-----------//
	// >first tuto
	void createDescriptorPool();
	void createDescriptorSets();
	void createDescriptorSet2();

	void updateUniformBuffer(uint32_t currentImage, const MyCamera& camera, const uint32_t& scalar, const glm::vec4& pos);
	void createUniformBuffers();
	void createNupdateUniformBuffers();
	void createDescriptorSetLayout(); // !! we create two descriptor set layouts

	glm::mat4 setPerspectiveProjection(float fovy, float aspect, float near, float far);

	VkDescriptorSetLayout descriptorSetLayout; //contains 1 global ubo
	VkDescriptorSetLayout descriptorSetLayout2; //contains 1 object ubo
	VkDescriptorSetLayout _drawImageDescriptorLayout; //contains 3 images

	VkDescriptorPool descriptorPool;
	std::vector<VkDescriptorSet> descriptorSets;
	VkDescriptorSet descriptorSets2;

	VkBuffer uniformBuffers2;
	VkDeviceMemory uniformBuffersMemory2;
	void* uniformBuffersMapped2;

	std::vector<VkBuffer> uniformBuffers;
	std::vector<VkDeviceMemory> uniformBuffersMemory;
	std::vector<void*> uniformBuffersMapped;

	// <first tuto
	//>--model loading
	void loadModel(const std::string& filepath, std::vector<uint32_t>& indices, std::vector<Vertex>& vert);
	
	std::vector<uint32_t> _indicesLight;
	std::vector<Vertex> _verticesLight;

	std::vector<uint32_t> _indicesObject;
	std::vector<Vertex> _verticesObject;
	
	void createVertexBuffer(const std::vector<Vertex>& vertices, VkBuffer&, VkDeviceMemory&);
	void createIndexBuffer(const std::vector<uint32_t>& indices, VkBuffer&, VkDeviceMemory&);

	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

	//<--model loading

	VkBuffer vertexBufferLight;
	VkDeviceMemory vertexBufferMemoryLight;
	VkBuffer indexBufferLight;
	VkDeviceMemory indexBufferMemoryLight;

	VkBuffer vertexBufferObject;
	VkDeviceMemory vertexBufferMemoryObject;
	VkBuffer indexBufferObject;
	VkDeviceMemory indexBufferMemoryObject;
	//<-----------LVE VULKAN TUTO-----------//
};

class KeyboardMovementController {
public:
	struct KeyMappings {
		int moveLeft = GLFW_KEY_A;
		int moveRight = GLFW_KEY_D;
		int moveForward = GLFW_KEY_W;
		int moveBackward = GLFW_KEY_S;
		int moveUp = GLFW_KEY_E;
		int moveDown = GLFW_KEY_Q;
		int lookLeft = GLFW_KEY_LEFT;
		int lookRight = GLFW_KEY_RIGHT;
		int lookUp = GLFW_KEY_UP;
		int lookDown = GLFW_KEY_DOWN;
	};

	void moveInPlaneXZ(GLFWwindow* window, float dt);

	KeyMappings keys{};
	float moveSpeed{ 10.f };
	float lookSpeed{ 1.5f };

	glm::vec3 translation{ 0.0f, -0.0f, -10.0f };
	glm::vec3 rotation{};
};