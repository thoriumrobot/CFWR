/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class SkipBufferedReader_slice {
    @Positive
  public static void method() throws IOException {
    @Positive
    BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(System.in));

    // :: error: (argument)
    @Positive
    bufferedReader.skip(-1);

    @Positive
    bufferedReader.skip(1);
    @Positive
  }

    private float __cfwr_aux657() {
        for (int __cfwr_i42 = 0; __cfwr_i42 < 4; __cfwr_i42++) {
            return null;
        }

        return ((-37.55 << 171) % ('s' ^ 546L));
        return null;
        return 78.09f;
    }
}