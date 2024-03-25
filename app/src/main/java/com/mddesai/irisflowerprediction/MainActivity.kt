package com.mddesai.irisflowerprediction

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import com.mddesai.irisflowerprediction.ml.IrisLiteModel
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)


        var input1=findViewById<EditText>(R.id.input1)
        var input2=findViewById<EditText>(R.id.input2)
        var input3=findViewById<EditText>(R.id.input3)
        var input4 = findViewById<EditText>(R.id.input4)
        var btn = findViewById<Button>(R.id.btn)
        var opText =findViewById<TextView>(R.id.opText)

btn.setOnClickListener(View.OnClickListener {
    hideKeyboard()
    val v1Text = input1.text.toString()
    val v2Text = input2.text.toString()
    val v3Text = input3.text.toString()
    val v4Text = input4.text.toString()

    if (v1Text.isEmpty() || v2Text.isEmpty() || v3Text.isEmpty() || v4Text.isEmpty()) {
        Toast.makeText(this, "Please fill all input fields", Toast.LENGTH_SHORT).show()
    } else  {
        val v1: Float = v1Text.toFloat()
        val v2: Float = v2Text.toFloat()
        val v3: Float = v3Text.toFloat()
        val v4: Float = v4Text.toFloat()


        // Proceed with inference

        var byteBuffer = ByteBuffer.allocateDirect(4 * 4)
        byteBuffer.putFloat(v1)
        byteBuffer.putFloat(v2)
        byteBuffer.putFloat(v3)
        byteBuffer.putFloat(v4)


        val model = IrisLiteModel.newInstance(this)

// Creates inputs for reference.
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 4), DataType.FLOAT32)
        inputFeature0.loadBuffer(byteBuffer)

        try {
// Runs model inference and gets result.
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray
            val predictedIndex = outputFeature0.indices.maxByOrNull { outputFeature0[it] } ?: -1


            val predicted: String = when (predictedIndex) {
                0 -> "Iris-setosa"
                1 -> "Iris-versicolor"
                2 -> "Iris-virginica"
                else -> "Unknown"
            }

            opText.setText("Your Predicted model:-" + predicted)

        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            // Releases model resources if no longer used.
            model.close()
        }
    }




})






    }
    private fun hideKeyboard() {
        val imm = getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager
        imm.hideSoftInputFromWindow(currentFocus?.windowToken, 0)
    }
}