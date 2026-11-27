import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'upload.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  Future<void> _pick(BuildContext context, ImageSource src) async {
    final img = await ImagePicker().pickImage(source: src, maxWidth: 1024, imageQuality: 90);
    if (img != null && context.mounted) {
      Navigator.push(context, MaterialPageRoute(builder: (_) => UploadScreen(path: img.path)));
    }
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(title: const Text('AgroDoc')),
        body: Center(
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            const Icon(Icons.eco, size: 96, color: Colors.green),
            const SizedBox(height: 12),
            const Text('Photograph one leaf on a plain background'),
            const SizedBox(height: 24),
            FilledButton.icon(
                onPressed: () => _pick(context, ImageSource.camera),
                icon: const Icon(Icons.photo_camera),
                label: const Text('Camera')),
            const SizedBox(height: 12),
            OutlinedButton.icon(
                onPressed: () => _pick(context, ImageSource.gallery),
                icon: const Icon(Icons.photo_library),
                label: const Text('Gallery')),
          ]),
        ),
      );
}
