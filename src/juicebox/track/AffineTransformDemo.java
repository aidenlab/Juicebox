package juicebox.track;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.AffineTransform;

/**
 * @author jrobinso
 *         Date: 1/8/13
 *         Time: 9:49 AM
 */
public class AffineTransformDemo {


    public static void main(String [] args) {

        JFrame frame = new JFrame();
        frame.setSize(500, 500);

        frame.getContentPane().setLayout(new BorderLayout());
        frame.getContentPane().add(new AffinePanel(), BorderLayout.CENTER);

        frame.setVisible(true);
    }


    static class AffinePanel extends JComponent {

        private static final long serialVersionUID = 2359945734983494428L;

        @Override
        protected void paintComponent(Graphics graphics) {

            Graphics2D g2D = (Graphics2D) graphics;

            int centerX = getWidth() / 2;
            int centerY = getWidth() / 2;

            g2D.fillRect(centerX, centerY-10, getWidth()/2, 10);
            g2D.fillRect(centerX + 100, centerY-20, getWidth()/2-200, 20);

            g2D.setColor(Color.blue);
            AffineTransform rotateTransform = new AffineTransform();
            rotateTransform.quadrantRotate(1);
            rotateTransform.scale(1, -1);
            g2D.transform(rotateTransform);
            g2D.fillRect(centerX, centerY-10, getWidth()/2, 10);
            g2D.fillRect(centerX + 100, centerY-20, getWidth()/2-200, 20);

//            g2D.setColor(Color.red);
//            rotateTransform = new AffineTransform();
//            rotateTransform.quadrantRotate(-1);
//            g2D.transform(rotateTransform);
//            g2D.drawLine(-centerX, -centerY, getWidth(), centerY);
//
//
//            g2D.setColor(Color.green);
//            rotateTransform = new AffineTransform();
//            rotateTransform.quadrantRotate(-1);
//            g2D.transform(rotateTransform);
//            g2D.drawLine(centerX, centerY, getWidth(), centerY);


        }
    }
}
