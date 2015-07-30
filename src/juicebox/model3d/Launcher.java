/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2015 Broad Institute, Aiden Lab
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.model3d;

import com.jogamp.opengl.*;
import com.jogamp.opengl.awt.GLCanvas;
import com.jogamp.opengl.util.FPSAnimator;

import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

/**
 * Created by muhammadsaadshamim on 7/24/15.
 */
public class Launcher extends Frame implements GLEventListener {


    private static final long serialVersionUID = -4436096213094040007L;

    public Launcher() {
        super("Basic JOGL Demo");

        setLayout(new BorderLayout());

        setSize(400, 400);
        setLocation(40, 40);

        // Need to set visible first before starting the rendering thread due
        // to a bug in JOGL. See JOGL Issue #54 for more information on this.
        // http://jogl.dev.java.net
        setVisible(true);

        setupJOGL();
    }

    //---------------------------------------------------------------
    // Methods defined by GLEventListener
    //---------------------------------------------------------------

    /**
     * Called by the drawable immediately after the OpenGL context is
     * initialized; the GLContext has already been made current when
     * this method is called.
     *
     * @param drawable The display context to render to
     */
    public void init(GLAutoDrawable drawable) {
        GL2 gl = drawable.getGL().getGL2();

        gl.glClearColor(0, 0, 0, 0);
        gl.glMatrixMode(GL2.GL_PROJECTION);
        gl.glLoadIdentity();
        gl.glOrtho(0, 1, 0, 1, -1, 1);
    }

    @Override
    public void dispose(GLAutoDrawable glAutoDrawable) {

    }

    /**
     * Called by the drawable when the surface resizes itself. Used to
     * reset the viewport dimensions.
     *
     * @param drawable The display context to render to
     */
    public void reshape(GLAutoDrawable drawable,
                        int x,
                        int y,
                        int width,
                        int height) {
    }

    /**
     * Called by the drawable when the display mode or the display device
     * associated with the GLDrawable has changed
     */
    public void displayChanged(GLAutoDrawable drawable,
                               boolean modeChanged,
                               boolean deviceChanged) {
    }

    /**
     * Called by the drawable to perform rendering by the client.
     *
     * @param drawable The display context to render to
     */
    public void display(GLAutoDrawable drawable) {
        GL2 gl = drawable.getGL().getGL2();

        gl.glClear(GL.GL_COLOR_BUFFER_BIT);

        gl.glBegin(GL.GL_TRIANGLES);

        gl.glColor3f(1, 0, 0);
        gl.glVertex3f(0.25f, 0.25f, 0);

        gl.glColor3f(0, 1, 0);
        gl.glVertex3f(0.5f, 0.25f, 0);

        gl.glColor3f(0, 0, 1);
        gl.glVertex3f(0.25f, 0.5f, 0);
        gl.glEnd();


        gl.glBegin(GL.GL_LINE_STRIP);
        float z = -10.0f;
        for (float angle = 0.0f; angle <= (2.0f * Math.PI) * 3.0f; angle += 0.1f) {
            float x = 50.0f * (float) (Math.sin(angle));
            float y = 50.0f * (float) (Math.cos(angle));

            // Specify the point and move the z value up a little
            gl.glVertex3f(x, y, z);
            z += 0.5f;
        }

        gl.glEnd();
        gl.glFlush();

    }

    //---------------------------------------------------------------
    // Local methods
    //---------------------------------------------------------------

    /**
     * Create the basics of the JOGL screen details.
     */
    private void setupJOGL() {
        GLCapabilities caps = new GLCapabilities(GLProfile.getDefault());
        caps.setDoubleBuffered(true);
        caps.setHardwareAccelerated(true);

        final GLCanvas canvas = new GLCanvas(caps);
        canvas.addGLEventListener(this);

        add(canvas, BorderLayout.CENTER);
        addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent windowevent) {
                Launcher.this.remove(canvas);
                Launcher.this.dispose();
            }
        });

        FPSAnimator anim = new FPSAnimator(canvas, 60);
        anim.start();
    }

}
