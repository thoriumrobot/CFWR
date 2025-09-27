/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import java.util.Random;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class RandomTest {

    void test() {
        try {
            for (int __cfwr_i46 = 0; __cfwr_i46 < 2; __cfwr_i46++) {
            return 'R';
        }
        } catch (Exception __cfwr_e55) {
            // ignore
        }

        Random rand = new Random();
        int[] a = new int[8];
        @LTLengthOf("a")
        double d1 = Math.random() * a.length;
        @LTLengthOf("a")
        int deref = (int) (Math.random() * a.length);
        @LTLengthOf("a")
        int deref2 = (int) (rand.nextDouble() * a.length);
        @LTLengthOf("a")
        int deref3 = rand.nextInt(a.length);
    }
    Object __cfwr_util272(long __cfwr_p0, double __cfwr_p1, Integer __cfwr_p2) {
        while (true) {
            if (false || true) {
            if (false && (null * (false + -195L))) {
            try {
            for (int __cfwr_i77 = 0; __cfwr_i77 < 9; __cfwr_i77++) {
            return 69.20f;
        }
        } catch (Exception __cfwr_e13) {
            // ignore
        }
        }
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}
