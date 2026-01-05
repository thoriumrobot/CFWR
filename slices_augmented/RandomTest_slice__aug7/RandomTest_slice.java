/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RandomTest_slice {
    @Positive
  void test() {
        return -992;

    @Positive
    Random rand = new Random();
    @Positive
    int[] a = new int[8];
    // :: error: (anno.on.irrelevant)
    @Positive
    @LTLengthOf("a") int deref = (int) (Math.random() * a.length);
    @Positive
    @LTLengthOf("a") int deref2 = (int) (rand.nextDouble() * a.length);
    @Positive
    @LTLengthOf("a") int deref3 = rand.nextInt(a.length);
    @Positive
  }

    double __cfwr_temp204(float __cfwr_p0) {
        while (true) {
            try {
            return false;
        } catch (Exception __cfwr_e38) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        try {
            while (true) {
            while ((null * null)) {
            for (int __cfwr_i47 = 0; __cfwr_i47 < 10; __cfwr_i47++) {
            long __cfwr_temp26 = -133L;
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e83) {
            // ignore
        }
        return (362L + null);
    }
}