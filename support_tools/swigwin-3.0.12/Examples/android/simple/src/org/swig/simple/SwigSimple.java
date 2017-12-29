package org.swig.simple;

import android.app.Activity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.ScrollView;
import android.text.method.ScrollingMovementMethod;

public class SwigSimple extends Activity
{
    TextView outputText = null;
    ScrollView scroller = null;

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        outputText = (TextView)findViewById(R.id.OutputText);
        outputText.setText("Press 'Run' to start...\n");
        outputText.setMovementMethod(new ScrollingMovementMethod());

        scroller = (ScrollView)findViewById(R.id.Scroller);
    }

    public void onRunButtonClick(View view)
    {
      outputText.append("Started...\n");
      nativeCall();
      outputText.append("Finished!\n");
      
      // Ensure scroll to end of text
      scroller.post(new Runnable() {
        public void run() {
          scroller.fullScroll(ScrollView.FOCUS_DOWN);
        }
      });
    }

    /** Calls into C/C++ code */
    public void nativeCall()
    {
      // Call our gcd() function
      
      int x = 42;
      int y = 105;
      int g = example.gcd(x,y);
      outputText.append("The greatest common divisor of " + x + " and " + y + " is " + g + "\n");

      // Manipulate the Foo global variable

      // Output its current value
      double foo = example.getFoo();
      outputText.append("Foo = " + foo + "\n");

      // Change its value
      example.setFoo(3.1415926);

      // See if the change took effect
      outputText.append("Foo = " + example.getFoo() + "\n");

      // Restore value
      example.setFoo(foo);
    }

    /** static constructor */
    static {
        System.loadLibrary("example");
    }
}

