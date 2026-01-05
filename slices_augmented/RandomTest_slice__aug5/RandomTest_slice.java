/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RandomTest_slice {
    @Positive
  void test() {
        if ((('q' * null) >> true) || false) {
            return 21L;
        }

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

    public static float __cfwr_temp145() {
        return -67.01f;
        return (false * 712L);
        int __cfwr_elem71 = -778;
        try {
            Integer __cfwr_elem93 = null;
        } catch (Exception __cfwr_e5) {
            // ignore
        }
        return -82.74f;
    }
    private int __cfwr_compute533(float __cfwr_p0) {
        while (true) {
            return null;
            break; // Prevent infinite loops
        }
        while (false) {
            return false;
            break; // Prevent infinite loops
        }
        return (null - (70.27 << 83.75f));
    }
}