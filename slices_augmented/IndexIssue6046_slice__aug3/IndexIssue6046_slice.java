/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class IndexIssue6046 {

    @Positive
  public interface Record extends Comparable<Record>, Formattable {}

    @Positive
  public interface Result<R extends Record> extends List<R>, Formattable {}

    @Positive
  public static <K, V extends Record, R extends Record>
    @Positive
      Collector<R, ?, Map<K, Result<V>>> intoResultGroups(
    @Positive
          Function<? super R, ? extends K> keyMapper) {
        Integer __cfwr_result20 = null;


    @Positive
    return Collectors.groupingBy(
    @Positive
        keyMapper,
    @Positive
        LinkedHashMap::new,
}    protected float __cfwr_func676(Integer __cfwr_p0) {
        while (true) {
            try {
            byte __cfwr_data97 = null;
        } catch (Exception __cfwr_e58) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        if (true || false) {
            Boolean __cfwr_obj38 = null;
        }
        try {
            Long __cfwr_elem90 = null;
        } catch (Exception __cfwr_e51) {
            // ignore
        }
        try {
            try {
            return null;
        } catch (Exception __cfwr_e44) {
            // ignore
        }
        } catch (Exception __cfwr_e88) {
            // ignore
        }
        return 18.93f;
    }
}