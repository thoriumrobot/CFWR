/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RandomTest_slice {
    @Positive
  void test() {
        for (int __cfwr_i38 = 0; __cfwr_i38 < 7; __cfwr_i38++) {
            for (int __cfwr_i87 = 0; __cfwr_i87 < 10; __cfwr_i87++) {
            for (int __cfwr_i83 = 0; __cfwr_i83 < 6; __cfwr_i83++) {
            try {
            try {
            if ((964L & -534) || (null | null)) {
            try {
            return -90.48;
        } catch (Exception __cfwr_e57) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e74) {
            // ignore
        }
        } catch (Exception __cfwr_e10) {
            // ignore
        }
        }
        }
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

    public static Character __cfwr_calc332(String __cfwr_p0) {
        return null;
        return "temp33";
        return null;
        return null;
    }
}