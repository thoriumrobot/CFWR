/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RandomTest_slice {
  void test() {
        short __cfwr_entry46 = null;

    Random rand = new Random();
    int[] a = new int[8];
    // :: error: (anno.on.irrelevant)
    @LTLengthOf("a") double d1 = Math.random() * a.length;
    @LTLengthOf("a") int deref = (int) (Math.random() * a.length);
    @LTLengthOf("a") int deref2 = (int) (rand.nextDouble() * a.length);
    @LTLengthOf("a") int deref3 = rand.nextInt(a.length);
  }

    protected static int __cfwr_proc533(Character __cfwr_p0, float __cfwr_p1, Boolean __cfwr_p2) {
        return null;
        return 906;
    }
    private static short __cfwr_compute213(int __cfwr_p0, Character __cfwr_p1, float __cfwr_p2) {
        return (677 & '5');
        if (false && false) {
            try {
            return -94.38;
        } catch (Exception __cfwr_e24) {
            // ignore
        }
        }
        if (((-88.89f & -885L) | ('5' * 'A')) && true) {
            try {
            for (int __cfwr_i23 = 0; __cfwr_i23 < 4; __cfwr_i23++) {
            return ((true & -687) + (null - 'C'));
        }
        } catch (Exception __cfwr_e98) {
            // ignore
        }
        }
        return (false + false);
    }
}