package org.swig.classexample;

import android.app.Activity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.ScrollView;
import android.text.method.ScrollingMovementMethod;

public class SwigClass extends Activity
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
      // ----- Object creation -----

      outputText.append( "Creating some objects:\n" );
      Circle c = new Circle(10);
      outputText.append( "    Created circle " + c + "\n");
      Square s = new Square(10);
      outputText.append( "    Created square " + s + "\n");

      // ----- Access a static member -----

      outputText.append( "\nA total of " + Shape.getNshapes() + " shapes were created\n" );

      // ----- Member data access -----

      // Notice how we can do this using functions specific to
      // the 'Circle' class.
      c.setX(20);
      c.setY(30);

      // Now use the same functions in the base class
      Shape shape = s;
      shape.setX(-10);
      shape.setY(5);

      outputText.append( "\nHere is their current position:\n" );
      outputText.append( "    Circle = (" + c.getX() + " " + c.getY() + ")\n" );
      outputText.append( "    Square = (" + s.getX() + " " + s.getY() + ")\n" );

      // ----- Call some methods -----

      outputText.append( "\nHere are some properties of the shapes:\n" );
      Shape[] shapes = {c,s};
      for (int i=0; i<shapes.length; i++)
      {
        outputText.append( "   " + shapes[i].toString() + "\n" );
        outputText.append( "        area      = " + shapes[i].area() + "\n" );
        outputText.append( "        perimeter = " + shapes[i].perimeter() + "\n" );
      }

      // Notice how the area() and perimeter() functions really
      // invoke the appropriate virtual method on each object.

      // ----- Delete everything -----

      outputText.append( "\nGuess I'll clean up now\n" );

      // Note: this invokes the virtual destructor
      // You could leave this to the garbage collector
      c.delete();
      s.delete();

      outputText.append( Shape.getNshapes() + " shapes remain\n" );
      outputText.append( "Goodbye\n" );
    }

    /** static constructor */
    static {
        System.loadLibrary("example");
    }
}
