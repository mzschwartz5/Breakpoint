Breakpoint is a project created by Daniel Gerhardt, Dineth Meegoda, Matt Schwartz, and Zixiao Wang. The goal is to combine PBMPM, a recent particle simulation method created by EA with fluid mesh shading using another 2024 research paper so as to make the simulation work in real time. 

Initially our plan was to incorporate the Mesh Mortal Kombat paper for the voxelization and destruction of soft bodies using PBMPM particles, but the scope of the project became too large.

## Milestone 1

The largest development for milestone 1 was the creation of the DirectX 12 engine. The project was created from scratch, so a lot of time and effort went in to putting together a solid foundation for the rest of the project that could scale for the many pipelines and compute passes required for combing PBMPM with mesh shading fluids. Additionally, compute passes, structured buffers for GPU data passing, and initial mesh shaders were set up for this milestone.

## Milestone 2

During milestone 2, PBMPM and PBD were implemented. PBMPM in 2D had some issues, mainly with volume loss and grid managemement, since the grid was not properly scaled for the camera setup. PBD was implemented with face and distance constraints for voxels, but after this milestone the work was sidelined to focus on getting PBMPM set up for 3D. The compute passes for mesh fluid shading were also created, and many optimizations were added to ensure the speed and optimality of the rendering of the fluids. 

## Milestone 3

For milestone 3, the 3D framework for PBMPM was created. Additionally emission and dynamic forces were added to the implementation of PBMPM, which aided in the fix for the grid bug present in milestone 2. The framework for mesh shading the fluids was completed, so all of the pieces for the finished product are in place. The setback is that there are issues with the movement of the 3D particles, and the mesh shading fluid setup needs to be adapted to have PBMPM particles as their input, which are the next steps for the project.

Helpful resources: 
DX12 basics, compointer: https://github.com/Ohjurot/D3D12Ez
